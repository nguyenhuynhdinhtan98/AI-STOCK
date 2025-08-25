"""
Refactored Vietnam Stock Analysis Toolkit
-----------------------------------------
- Tách lớp rõ ràng: DataFetcher, FeatureEngineer, Scorer, PromptBuilder, AIAssistant, Analyzer
- Chuẩn hoá logging, type hints, dataclasses
- Giảm I/O thừa, xử lý thiếu API key an toàn (bỏ qua phần AI thay vì raise)
- Sử dụng ta.trend.MACD class ổn định hơn
- Sửa các lỗi fillna không gán, kiểm tra None với Ichimoku/RS
- CLI tiện dụng qua argparse (đa mã, tuỳ chọn bật/tắt AI, min market cap, ngày)
"""
from __future__ import annotations

import os
import json
import logging
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# 3rd party libs
import ta
from dotenv import load_dotenv
from vnstock.explorer.vci import Quote, Finance, Company
from vnstock import Screener

# Optional AI deps
try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None

# -------------------- Logging --------------------
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                    handlers=[
                        logging.FileHandler("stock_analysis.log", encoding="utf-8"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger("vnstocks")

# -------------------- Config --------------------
@dataclass(frozen=True)
class AppConfig:
    start_date: str
    end_date: str
    data_dir: str = "vnstocks_data"
    vni_symbol: str = "VNINDEX"
    min_market_cap: int = 500  # đơn vị: tỷ VND (tuỳ nguồn vnstock)
    use_gemini: bool = True
    use_openrouter: bool = True

    @staticmethod
    def default() -> "AppConfig":
        today = datetime.today()
        return AppConfig(
            start_date=(today - timedelta(days=365 * 15)).strftime("%Y-%m-%d"),
            end_date=today.strftime("%Y-%m-%d"),
        )

@dataclass(frozen=True)
class Weights:
    ma: float = 35.0
    rsi: float = 14.0
    macd: float = 14.0
    ichimoku: float = 14.0
    volume: float = 14.0
    rs: float = 14.0
    bb: float = 7.0

# -------------------- Utils --------------------
def safe_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            if np.isnan(val) or np.isinf(val):
                return None
            return float(val)
        return float(str(val).replace(",", ""))
    except Exception:
        return None


def validate_dataframe(df: Optional[pd.DataFrame], required_columns: Optional[List[str]] = None) -> bool:
    if df is None or df.empty:
        return False
    if required_columns:
        return all(col in df.columns for col in required_columns)
    return True


# -------------------- Data Layer --------------------
class DataFetcher:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        os.makedirs(cfg.data_dir, exist_ok=True)
        self._vni_cache: Optional[pd.DataFrame] = None

    def get_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Lấy dữ liệu lịch sử giá cổ phiếu từ VCI và lưu CSV."""
        try:
            logger.info("Lấy dữ liệu %s", symbol)
            df = Quote(symbol=symbol).history(start=self.cfg.start_date,
                                              end=self.cfg.end_date,
                                              interval="1D")
            if not validate_dataframe(df, ["time", "open", "high", "low", "close", "volume"]):
                logger.warning("Không có dữ liệu cho %s", symbol)
                return None
            df = df.rename(columns={
                "time": "Date", "open": "Open", "high": "High",
                "low": "Low", "close": "Close", "volume": "Volume"
            })
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            path = os.path.join(self.cfg.data_dir, f"{symbol}_data.csv")
            df.to_csv(path, encoding="utf-8-sig")
            return df
        except Exception as e:
            logger.error("Lỗi get_stock_data(%s): %s", symbol, e)
            return None

    def get_market_data(self) -> Optional[pd.DataFrame]:
        if self._vni_cache is not None:
            return self._vni_cache
        try:
            logger.info("Lấy dữ liệu %s", self.cfg.vni_symbol)
            df = Quote(symbol=self.cfg.vni_symbol).history(start=self.cfg.start_date,
                                                           end=self.cfg.end_date,
                                                           interval="1D")
            if not validate_dataframe(df, ["time", "open", "high", "low", "close", "volume"]):
                logger.warning("Không lấy được dữ liệu %s", self.cfg.vni_symbol)
                return None
            df = df.rename(columns={
                "time": "Date", "open": "Open", "high": "High",
                "low": "Low", "close": "Close", "volume": "Volume"
            })
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            path = os.path.join(self.cfg.data_dir, f"{self.cfg.vni_symbol}_data.csv")
            df.to_csv(path, encoding="utf-8-sig")
            self._vni_cache = df
            return df
        except Exception as e:
            logger.error("Lỗi get_market_data(): %s", e)
            return None

    def get_financial_data(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            logger.info("Lấy BCTC %s (quarter)", symbol)
            fin = Finance(symbol=symbol, period="quarter")
            df_ratio = fin.ratio(period="quarter")
            df_bs = fin.balance_sheet(period="quarter")
            df_is = fin.income_statement(period="quarter")
            df_cf = fin.cash_flow(period="quarter")

            def flatten(df: pd.DataFrame) -> pd.DataFrame:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ["_".join([c for c in col if c]) for col in df.columns]
                return df

            def standardize(df: pd.DataFrame) -> pd.DataFrame:
                mapping = {"Meta_ticker": "ticker", "Meta_yearReport": "yearReport", "Meta_lengthReport": "lengthReport"}
                return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

            df_ratio = standardize(flatten(df_ratio))
            financial = df_bs.merge(df_is, on=["yearReport", "lengthReport", "ticker"], how="outer") \
                              .merge(df_cf, on=["yearReport", "lengthReport", "ticker"], how="outer") \
                              .merge(df_ratio, on=["yearReport", "lengthReport", "ticker"], how="outer")
            financial = financial.rename(columns={"ticker": "Symbol", "yearReport": "Year", "lengthReport": "Quarter"})
            financial = financial.tail(20)
            path = os.path.join(self.cfg.data_dir, f"{symbol}_financial_statements.csv")
            financial.to_csv(path, index=False, encoding="utf-8-sig")
            return financial
        except Exception as e:
            logger.error("Lỗi get_financial_data(%s): %s", symbol, e)
            return None

    def get_company_info(self, symbol: str) -> str:
        try:
            logger.info("Lấy thông tin công ty %s", symbol)
            comp = Company(symbol)
            sections = {
                "OVERVIEW": comp.overview(),
                "SHAREHOLDERS": comp.shareholders(),
                "OFFICERS": comp.officers(filter_by="working"),
                "EVENTS": comp.events(),
                "NEWS": comp.news(),
                "REPORTS": comp.reports(),
                "TRADING STATS": comp.trading_stats(),
                "RATIO SUMMARY": comp.ratio_summary(),
            }
            lines: List[str] = []
            for name, data in sections.items():
                lines.append(f"=== {name} ===")
                if isinstance(data, pd.DataFrame):
                    lines.append(data.to_string() if not data.empty else "Không có dữ liệu")
                elif isinstance(data, dict):
                    lines.append(json.dumps(data, ensure_ascii=False, indent=2) if data else "Không có dữ liệu")
                elif data is not None:
                    lines.append(str(data))
                else:
                    lines.append("Không có dữ liệu")
                lines.append("")
            text = "\n".join(lines)
            path = os.path.join(self.cfg.data_dir, f"{symbol}_company_info.txt")
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write(text)
            return text
        except Exception as e:
            logger.error("Lỗi get_company_info(%s): %s", symbol, e)
            return f"Không thể lấy thông tin công ty ({e})"

    def filter_stocks_low_pe_high_cap(self, min_market_cap: Optional[int] = None) -> Optional[pd.DataFrame]:
        try:
            mc = min_market_cap if min_market_cap is not None else self.cfg.min_market_cap
            logger.info("Lọc cổ phiếu: P/E<20 & vốn hoá >= %s", mc)
            df = Screener().stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=5000)
            if not validate_dataframe(df):
                logger.error("Không lấy được danh sách cổ phiếu")
                return None
            cond1 = df["market_cap"] >= mc
            cond2 = ((df["pe"] > 0) & (df["pe"] < 20)) | pd.isna(df["pe"])  # cho phép trống
            cond3 = (df["pb"] > 0) | pd.isna(df["pb"])  # không quá khắt khe
            cond4 = (df["last_quarter_revenue_growth"] >= 0) | pd.isna(df["last_quarter_revenue_growth"])  # >=0
            cond5 = (df["second_quarter_revenue_growth"] >= 0) | pd.isna(df["second_quarter_revenue_growth"])  # >=0
            cond6 = (df["last_quarter_profit_growth"] >= 0) | pd.isna(df["last_quarter_profit_growth"])  # >=0
            cond7 = (df["second_quarter_profit_growth"] >= 0) | pd.isna(df["second_quarter_profit_growth"])  # >=0
            cond8 = ((df["peg_forward"] >= 0) & (df["peg_forward"] < 1)) | pd.isna(df["peg_forward"])  # tăng trưởng hợp lý
            cond9 = ((df["peg_trailing"] >= 0) & (df["peg_trailing"] < 1)) | pd.isna(df["peg_trailing"])  # tăng trưởng hợp lý
            cond10 = (df["revenue_growth_1y"] >= 0) | pd.isna(df["revenue_growth_1y"])  # >=0
            cond11 = (df["eps_growth_1y"] >= 0) | pd.isna(df["eps_growth_1y"])  # >=0
            final = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8 & cond9 & cond10 & cond11
            filtered = df[final].copy()
            base_path = os.path.join(self.cfg.data_dir, "market_filtered.csv")
            pe_path = os.path.join(self.cfg.data_dir, "market_filtered_pe.csv")
            df[cond1].to_csv(base_path, index=False, encoding="utf-8-sig")
            filtered.to_csv(pe_path, index=False, encoding="utf-8-sig")
            logger.info("Đã lưu list lọc: %s hàng", len(filtered))
            return filtered
        except Exception as e:
            logger.error("Lỗi filter_stocks_low_pe_high_cap: %s", e)
            return None


# -------------------- Feature Layer --------------------
class FeatureEngineer:
    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        if not validate_dataframe(df):
            return df
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].ffill().bfill()
        df["returns"] = df["Close"].pct_change()
        df["volatility"] = df["returns"].rolling(10).std()
        return df

    @staticmethod
    def create_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if not validate_dataframe(df, ["Close", "High", "Low", "Volume"]):
            return df
        df = df.copy()
        # SMA
        for w in [10, 20, 50, 200]:
            df[f"SMA_{w}"] = ta.trend.sma_indicator(df["Close"], window=w)
        # RSI
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
        # MACD (ổn định)
        macd = ta.trend.MACD(close=df["Close"])  # default 12, 26, 9
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = macd.macd_diff()
        # Bollinger
        bb = ta.volatility.BollingerBands(close=df["Close"])  # default 20, 2
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Middle"] = bb.bollinger_mavg()
        df["BB_Lower"] = bb.bollinger_lband()
        # Volume SMA
        for w in [20, 50]:
            df[f"Volume_MA_{w}"] = ta.trend.sma_indicator(df["Volume"], window=w)
        # Ichimoku
        ichi = ta.trend.IchimokuIndicator(high=df["High"], low=df["Low"], window1=9, window2=26, window3=52)
        df["ichimoku_tenkan_sen"] = ichi.ichimoku_conversion_line()
        df["ichimoku_kijun_sen"] = ichi.ichimoku_base_line()
        df["ichimoku_senkou_span_a"] = ichi.ichimoku_a()
        df["ichimoku_senkou_span_b"] = ichi.ichimoku_b()
        df["ichimoku_chikou_span"] = df["Close"].shift(26)
        return df

    @staticmethod
    def relative_strength(df_stock: pd.DataFrame, df_index: pd.DataFrame) -> pd.DataFrame:
        if not (validate_dataframe(df_stock) and validate_dataframe(df_index)):
            return df_stock
        try:
            merged = df_stock[["Close"]].join(df_index[["Close"]].rename(columns={"Close": "Index_Close"}), how="inner")
            if merged.empty or merged["Index_Close"].isna().all():
                logger.warning("Không có VNI để tính RS; dùng mặc định")
                for col, v in {
                    "RS": 1.0,
                    "RS_Point": 0.0,
                    "RS_SMA_10": 1.0,
                    "RS_SMA_20": 1.0,
                    "RS_SMA_50": 1.0,
                    "RS_SMA_200": 1.0,
                    "RS_Point_SMA_10": 0.0,
                    "RS_Point_SMA_20": 0.0,
                    "RS_Point_SMA_50": 0.0,
                    "RS_Point_SMA_200": 0.0,
                }.items():
                    df_stock[col] = v
                return df_stock
            merged["RS"] = merged["Close"] / merged["Index_Close"]
            roc_63 = ta.momentum.roc(merged["Close"], window=63)
            roc_126 = ta.momentum.roc(merged["Close"], window=126)
            roc_189 = ta.momentum.roc(merged["Close"], window=189)
            roc_252 = ta.momentum.roc(merged["Close"], window=252)
            merged["RS_Point"] = (roc_63 * 0.4 + roc_126 * 0.2 + roc_189 * 0.2 + roc_252 * 0.2) * 100
            for w in [10, 20, 50, 200]:
                merged[f"RS_SMA_{w}"] = ta.trend.sma_indicator(merged["RS"], window=w)
                merged[f"RS_Point_SMA_{w}"] = ta.trend.sma_indicator(merged["RS_Point"], window=w)
            cols = [
                "RS", "RS_Point", "RS_SMA_10", "RS_SMA_20", "RS_SMA_50", "RS_SMA_200",
                "RS_Point_SMA_10", "RS_Point_SMA_20", "RS_Point_SMA_50", "RS_Point_SMA_200",
            ]
            df_stock = df_stock.join(merged[cols], how="left")
            for c in cols:
                if "RS_Point" in c:
                    df_stock[c] = df_stock[c].fillna(0.0)
                else:
                    df_stock[c] = df_stock[c].fillna(1.0)
            return df_stock
        except Exception as e:
            logger.error("Lỗi relative_strength: %s", e)
            return df_stock

    @staticmethod
    def rs_snapshot_from_file(symbol: str, data_dir: str) -> Tuple[float, float, float, float]:
        """Đọc RS 3d/1m/3m/1y từ market_filtered.csv nếu có; mặc định 1.0."""
        try:
            file_path = os.path.join(data_dir, "market_filtered.csv")
            if not os.path.exists(file_path):
                return 1.0, 1.0, 1.0, 1.0
            market_df = pd.read_csv(file_path)
            if "ticker" not in market_df.columns:
                return 1.0, 1.0, 1.0, 1.0
            row = market_df[market_df["ticker"].str.upper() == symbol.upper()]
            if row.empty:
                return 1.0, 1.0, 1.0, 1.0
            out_path = os.path.join(data_dir, f"{symbol}_infor.csv")
            row.to_csv(out_path, index=False, encoding="utf-8-sig")
            v3d = row.get("relative_strength_3d", pd.Series([1.0])).iloc[0]
            v1m = row.get("rel_strength_1m", pd.Series([1.0])).iloc[0]
            v3m = row.get("rel_strength_3m", pd.Series([1.0])).iloc[0]
            v1y = row.get("rel_strength_1y", pd.Series([1.0])).iloc[0]
            return float(v3d), float(v1m), float(v3m), float(v1y)
        except Exception as e:
            logger.error("Lỗi rs_snapshot_from_file: %s", e)
            return 1.0, 1.0, 1.0, 1.0


# -------------------- Scoring --------------------
class Scorer:
    def __init__(self, weights: Weights, cfg: AppConfig) -> None:
        self.w = weights
        self.cfg = cfg

    @staticmethod
    def _empty_signal() -> Dict[str, Any]:
        return {
            "signal": "LỖI", "score": 50.0, "current_price": 0.0,
            "rsi_value": None, "ma10": None, "ma20": None, "ma50": None, "ma200": None,
            "rs": 1.0, "rs_point": 0.0, "recommendation": "KHÔNG XÁC ĐỊNH",
            "open": None, "high": None, "low": None, "volume": None,
            "volume_ma_20": None, "volume_ma_50": None,
            "macd": None, "macd_signal": None, "macd_hist": None,
            "bb_upper": None, "bb_lower": None,
            "ichimoku_tenkan_sen": None, "ichimoku_kijun_sen": None,
            "ichimoku_senkou_span_a": None, "ichimoku_senkou_span_b": None,
            "ichimoku_chikou_span": None,
            "rs_sma_10": None, "rs_sma_20": None, "rs_sma_50": None, "rs_sma_200": None,
            "rs_point_sma_10": None, "rs_point_sma_20": None, "rs_point_sma_50": None, "rs_point_sma_200": None,
            "relative_strength_3d": None, "relative_strength_1m": None,
            "relative_strength_3m": None, "relative_strength_1y": None,
            "forecast_dates": [], "forecast_prices": [], "forecast_plot_path": "",
        }

    def score(self, df: pd.DataFrame, symbol: str) -> Tuple[float, Dict[str, Any]]:
        if not validate_dataframe(df):
            return 50.0, self._empty_signal()
        try:
            last = df.iloc[-1]
            current_price = safe_float(last.get("Close"))
            if current_price is None:
                return 50.0, self._empty_signal()
            # Indicators snapshot
            ind = {
                "rsi": safe_float(last.get("RSI")),
                "ma10": safe_float(last.get("SMA_10", current_price)),
                "ma20": safe_float(last.get("SMA_20", current_price)),
                "ma50": safe_float(last.get("SMA_50", current_price)),
                "ma200": safe_float(last.get("SMA_200", current_price)),
                "macd": safe_float(last.get("MACD")),
                "macd_signal": safe_float(last.get("MACD_Signal")),
                "macd_hist": safe_float(last.get("MACD_Hist")),
                "bb_upper": safe_float(last.get("BB_Upper")),
                "bb_lower": safe_float(last.get("BB_Lower")),
                "vol_ma20": safe_float(last.get("Volume_MA_20", df["Volume"].rolling(20).mean().iloc[-1] if len(df) >= 20 else None)),
                "vol_ma50": safe_float(last.get("Volume_MA_50", df["Volume"].rolling(50).mean().iloc[-1] if len(df) >= 50 else None)),
            }
            # Ichimoku snapshot
            ichi = {
                "tenkan": safe_float(last.get("ichimoku_tenkan_sen")),
                "kijun": safe_float(last.get("ichimoku_kijun_sen")),
                "a": safe_float(last.get("ichimoku_senkou_span_a")),
                "b": safe_float(last.get("ichimoku_senkou_span_b")),
                "chikou": safe_float(last.get("ichimoku_chikou_span")),
            }
            # RS snapshot inline + file-based snapshot
            rs = safe_float(last.get("RS", 1.0)) if symbol.upper() != self.cfg.vni_symbol else 1.0
            rs_point = safe_float(last.get("RS_Point", 0.0)) if symbol.upper() != self.cfg.vni_symbol else 0.0
            rs3d, rs1m, rs3m, rsy = FeatureEngineer.rs_snapshot_from_file(symbol, self.cfg.data_dir)

            score = 50.0
            # MA - 35%
            ma_score = 0.0
            for key in ("ma10", "ma20", "ma50", "ma200"):
                v = ind.get(key)
                if v is not None and current_price > v:
                    ma_score += self.w.ma * 0.1  # 4 * 3.5 = 14 -> chuẩn hoá theo 35%/10
            ma_list = [ind.get("ma10"), ind.get("ma20"), ind.get("ma50"), ind.get("ma200")]
            if all(v is not None for v in ma_list):
                asc = all(ma_list[i] >= ma_list[i+1] for i in range(3))
                desc = all(ma_list[i] <= ma_list[i+1] for i in range(3))
                if asc:
                    ma_score += self.w.ma * 0.1
                elif desc:
                    ma_score -= self.w.ma * 0.1
            score += ma_score

            # RSI - 14%
            rsi_score = 0.0
            rsi = ind.get("rsi")
            if rsi is not None:
                if rsi < 30: rsi_score += self.w.rsi
                elif 30 <= rsi < 40: rsi_score += self.w.rsi * (10/14)
                elif 40 <= rsi < 50: rsi_score += self.w.rsi * (7/14)
                elif 50 <= rsi < 60: rsi_score += self.w.rsi * (3.5/14)
                elif 60 <= rsi < 70: rsi_score -= self.w.rsi * (3.5/14)
                elif 70 <= rsi < 80: rsi_score -= self.w.rsi * (7/14)
                else: rsi_score -= self.w.rsi
            score += rsi_score

            # MACD - 14%
            macd_score = 0.0
            if ind["macd"] is not None and ind["macd_signal"] is not None and ind["macd_hist"] is not None:
                if ind["macd"] > ind["macd_signal"] and ind["macd_hist"] > 0:
                    macd_score += self.w.macd * 0.5
                elif ind["macd"] < ind["macd_signal"] and ind["macd_hist"] < 0:
                    macd_score -= self.w.macd * 0.5
                if len(df) > 1:
                    prev_hist = safe_float(df["MACD_Hist"].iloc[-2])
                    if prev_hist is not None:
                        macd_score += (self.w.macd * 0.25) if ind["macd_hist"] > prev_hist else -(self.w.macd * 0.25)
                    prev_macd = safe_float(df["MACD"].iloc[-2])
                    prev_sig = safe_float(df["MACD_Signal"].iloc[-2])
                    if prev_macd is not None and prev_sig is not None:
                        crossed_up = ind["macd"] > ind["macd_signal"] and prev_macd <= prev_sig
                        crossed_dn = ind["macd"] < ind["macd_signal"] and prev_macd >= prev_sig
                        if crossed_up: macd_score += self.w.macd * 0.25
                        if crossed_dn: macd_score -= self.w.macd * 0.25
            score += macd_score

            # Ichimoku - 14%
            ichi_score = 0.0
            if ichi["a"] is not None and ichi["b"] is not None:
                top = max(ichi["a"], ichi["b"])
                bot = min(ichi["a"], ichi["b"])
                if current_price > top:
                    ichi_score += self.w.ichimoku
                elif current_price < bot:
                    ichi_score -= self.w.ichimoku
            score += ichi_score

            # Volume - 14%
            vol_score = 0.0
            cur_vol = safe_float(last.get("Volume"))
            if cur_vol is not None:
                if ind["vol_ma20"] and ind["vol_ma20"] > 0:
                    ratio20 = cur_vol / ind["vol_ma20"]
                    if ratio20 > 2.0: vol_score += 4
                    elif ratio20 > 1.5: vol_score += 3
                    elif ratio20 > 1.0: vol_score += 1
                    elif ratio20 < 0.5: vol_score -= 2
                if ind["vol_ma50"] and ind["vol_ma50"] > 0:
                    ratio50 = cur_vol / ind["vol_ma50"]
                    if ratio50 > 2.0: vol_score += 3
                    elif ratio50 > 1.5: vol_score += 2
                    elif ratio50 > 1.0: vol_score += 1
                    elif ratio50 < 0.5: vol_score -= 1
                if len(df) > 2:
                    v1 = safe_float(df["Volume"].iloc[-2])
                    v2 = safe_float(df["Volume"].iloc[-3])
                    if v1 is not None and v2 is not None:
                        if cur_vol > v1 > v2:
                            vol_score += 4 if cur_vol / v2 > 1.5 else 2
                        elif cur_vol < v1 < v2:
                            vol_score -= 4 if cur_vol / v2 < 0.7 else 2
                vol_score = max(min(vol_score, self.w.volume), -self.w.volume)
            score += vol_score

            # RS - 14% (không áp dụng cho VNINDEX)
            if symbol.upper() != self.cfg.vni_symbol:
                rs_score = 0.0
                rs_sma10 = safe_float(last.get("RS_SMA_10", rs))
                rs_sma50 = safe_float(last.get("RS_SMA_50", rs))
                rspt_sma20 = safe_float(last.get("RS_Point_SMA_20", 0.0))
                if rs is not None and rs_sma10 is not None:
                    rs_score += (self.w.rs * 0.25) if rs > rs_sma10 else -(self.w.rs * 0.25)
                if rs is not None and rs_sma50 is not None:
                    rs_score += (self.w.rs * 0.25) if rs > rs_sma50 else -(self.w.rs * 0.25)
                if rs_point is not None and rspt_sma20 is not None:
                    rs_score += (self.w.rs * 0.25) if rs_point > rspt_sma20 else -(self.w.rs * 0.25)
                if rs_point is not None:
                    if rs_point > 1.0: rs_score += self.w.rs * 0.25
                    elif rs_point < -1.0: rs_score -= self.w.rs * 0.25
                score += rs_score

            # Bollinger - 7%
            bb_score = 0.0
            if ind["bb_upper"] is not None and ind["bb_lower"] is not None and ind["bb_upper"] > ind["bb_lower"]:
                width = ind["bb_upper"] - ind["bb_lower"]
                to_upper = (ind["bb_upper"] - current_price) / width
                to_lower = (current_price - ind["bb_lower"]) / width
                if to_lower < 0.15: bb_score += self.w.bb
                elif to_lower < 0.30: bb_score += self.w.bb * 0.5
                if to_upper < 0.15: bb_score -= self.w.bb
                elif to_upper < 0.30: bb_score -= self.w.bb * 0.5
                if len(df) > 1:
                    prev_u = safe_float(df["BB_Upper"].iloc[-2])
                    prev_l = safe_float(df["BB_Lower"].iloc[-2])
                    if prev_u is not None and prev_l is not None and (prev_u - prev_l) > 0:
                        prev_w = prev_u - prev_l
                        if width > prev_w * 1.1: bb_score -= self.w.bb * 0.25
                        elif width < prev_w * 0.9: bb_score += self.w.bb * 0.25
            score += bb_score

            # Clamp 0..100
            score = float(max(0.0, min(100.0, score)))

            # Signal mapping
            if score >= 80:
                signal, reco = "MUA MẠNH", "MUA MẠNH"
            elif score >= 65:
                signal, reco = "MUA", "MUA"
            elif score >= 55:
                signal, reco = "TĂNG MẠNH", "GIỮ - TĂNG"
            elif score >= 45:
                signal, reco = "TRUNG LẬP", "GIỮ"
            elif score >= 35:
                signal, reco = "GIẢM MẠNH", "GIỮ - GIẢM"
            elif score >= 20:
                signal, reco = "BÁN", "BÁN"
            else:
                signal, reco = "BÁN MẠNH", "BÁN MẠNH"

            out = {
                "signal": signal,
                "score": score,
                "current_price": current_price,
                "rsi_value": ind["rsi"],
                "ma10": ind["ma10"], "ma20": ind["ma20"], "ma50": ind["ma50"], "ma200": ind["ma200"],
                "rs": rs, "rs_point": rs_point,
                "recommendation": reco,
                "open": safe_float(last.get("Open")),
                "high": safe_float(last.get("High")),
                "low": safe_float(last.get("Low")),
                "volume": cur_vol,
                "volume_ma_20": ind["vol_ma20"], "volume_ma_50": ind["vol_ma50"],
                "macd": ind["macd"], "macd_signal": ind["macd_signal"], "macd_hist": ind["macd_hist"],
                "bb_upper": ind["bb_upper"], "bb_lower": ind["bb_lower"],
                "ichimoku_tenkan_sen": ichi["tenkan"],
                "ichimoku_kijun_sen": ichi["kijun"],
                "ichimoku_senkou_span_a": ichi["a"],
                "ichimoku_senkou_span_b": ichi["b"],
                "ichimoku_chikou_span": ichi["chikou"],
                "rs_sma_10": safe_float(last.get("RS_SMA_10")),
                "rs_sma_20": safe_float(last.get("RS_SMA_20")),
                "rs_sma_50": safe_float(last.get("RS_SMA_50")),
                "rs_sma_200": safe_float(last.get("RS_SMA_200")),
                "rs_point_sma_10": safe_float(last.get("RS_Point_SMA_10")),
                "rs_point_sma_20": safe_float(last.get("RS_Point_SMA_20")),
                "rs_point_sma_50": safe_float(last.get("RS_Point_SMA_50")),
                "rs_point_sma_200": safe_float(last.get("RS_Point_SMA_200")),
                "relative_strength_3d": rs3d, "relative_strength_1m": rs1m,
                "relative_strength_3m": rs3m, "relative_strength_1y": rsy,
                "forecast_dates": [], "forecast_prices": [], "forecast_plot_path": "",
            }
            return score, out
        except Exception as e:
            logger.error("Lỗi Scorer.score(%s): %s", symbol, e)
            return 50.0, self._empty_signal()


# -------------------- Prompt Builder --------------------
class PromptBuilder:
    @staticmethod
    def _fmt(v: Any) -> str:
        num = safe_float(v)
        if num is None:
            return "N/A"
        if abs(num) >= 1e9:
            return f"{num / 1e9:.2f}B"
        if abs(num) >= 1e6:
            return f"{num / 1e6:.2f}M"
        if abs(num) >= 1e3:
            return f"{num / 1e3:.2f}K"
        return f"{num:.2f}"

    @classmethod
    def stock_prompt(cls, symbol: str, current_price: float, technical: Dict[str, Any],
                     trading_signal: Dict[str, Any], financial: Optional[pd.DataFrame],
                     company_info: str, history_text: str, info_text: str, market_text: str) -> str:
        rsi = technical.get("rsi")
        ma = technical.get("ma", {})
        bb = technical.get("bollinger_bands", {})
        macd = technical.get("macd", {})
        ichi = technical.get("ichimoku", {})
        vol = technical.get("volume", {})
        prompt = f"""
YÊU CẦU PHÂN TÍCH CHUYÊN SÂU\nMÃ: {symbol}\nGIÁ HIỆN TẠI: {cls._fmt(current_price)} VND
1) XUNG LƯỢNG\n- RSI(14): {cls._fmt(rsi)}\n- MACD: {cls._fmt(macd.get('macd'))} | Signal: {cls._fmt(macd.get('signal'))} | Hist: {cls._fmt(macd.get('histogram'))}
2) TRUNG BÌNH GIÁ\n- MA10: {cls._fmt(ma.get('ma10'))} | MA20: {cls._fmt(ma.get('ma20'))} | MA50: {cls._fmt(ma.get('ma50'))} | MA200: {cls._fmt(ma.get('ma200'))}
3) BOLLINGER\n- Upper: {cls._fmt(bb.get('upper'))} | Lower: {cls._fmt(bb.get('lower'))}
4) ICHIMOKU\n- Tenkan: {cls._fmt(ichi.get('tenkan'))} | Kijun: {cls._fmt(ichi.get('kijun'))} | A: {cls._fmt(ichi.get('senkou_a'))} | B: {cls._fmt(ichi.get('senkou_b'))} | Chikou: {cls._fmt(ichi.get('chikou'))}
5) VOLUME\n- Cur: {cls._fmt(vol.get('current'))} | MA20: {cls._fmt(vol.get('ma20'))}
6) RS \n- 3D: {cls._fmt(trading_signal.get('relative_strength_3d'))} | 1M: {cls._fmt(trading_signal.get('relative_strength_1m'))} | 3M: {cls._fmt(trading_signal.get('relative_strength_3m'))} | 1Y: {cls._fmt(trading_signal.get('relative_strength_1y'))}
"""
        if financial is not None and not financial.empty:
            prompt += f"\nBÁO CÁO TÀI CHÍNH:\n{financial.to_string(index=False)}\n"
        else:
            prompt += "\nKHÔNG CÓ DỮ LIỆU BÁO CÁO TÀI CHÍNH\n"
        prompt += f"""
DỮ LIỆU LỊCH SỬ GIÁ:\n{history_text}\n\nTHÔNG TIN CÔNG TY:\n{company_info}\n\nTHÔNG TIN CHUNG TỪ TCBS:\n{info_text}\n\nTHÔNG TIN TOÀN BỘ CỔ PHIẾU THỊ TRƯỜNG CÓ PE<20 & TĂNG TRƯỞNG:\n{market_text}
[Hãy phân tích theo Wyckoff/VSA/Minervini/Canslim + cơ bản + định giá + chiến lược & rủi ro. Kết luận MUA/MUA MẠNH/GIỮ/BÁN/BÁN MẠNH, cho điểm 1–10 và tóm tắt 2–3 câu.]
"""
        return prompt

    @classmethod
    def vnindex_prompt(cls, symbol: str, current_price: float, technical: Dict[str, Any],
                       history_text: str, market_text: str) -> str:
        rsi = technical.get("rsi")
        ma = technical.get("ma", {})
        bb = technical.get("bollinger_bands", {})
        macd = technical.get("macd", {})
        ichi = technical.get("ichimoku", {})
        vol = technical.get("volume", {})
        prompt = f"""
VNINDEX PHÂN TÍCH TỔNG HỢP\nCHỈ SỐ: {symbol} | ĐIỂM: {cls._fmt(current_price)}\nRSI: {cls._fmt(rsi)} | MACD: {cls._fmt(macd.get('macd'))}/{cls._fmt(macd.get('signal'))}/{cls._fmt(macd.get('histogram'))}\nMA: 10={cls._fmt(ma.get('ma10'))}, 20={cls._fmt(ma.get('ma20'))}, 50={cls._fmt(ma.get('ma50'))}, 200={cls._fmt(ma.get('ma200'))}\nBB: U={cls._fmt(bb.get('upper'))}, L={cls._fmt(bb.get('lower'))}\nICHIMOKU: T={cls._fmt(ichi.get('tenkan'))}, K={cls._fmt(ichi.get('kijun'))}, A={cls._fmt(ichi.get('senkou_a'))}, B={cls._fmt(ichi.get('senkou_b'))}, C={cls._fmt(ichi.get('chikou'))}\nVOL: Cur={cls._fmt(vol.get('current'))} MA20={cls._fmt(vol.get('ma20'))}
DỮ LIỆU LỊCH SỬ:\n{history_text}\n\nPE-Filter Market:\n{market_text}\n[Hãy phân tích VSA/VPA/Wyckoff/Canslim + kịch bản 1–2 tuần + chiến lược vị thế/SL/TP + rủi ro.]
"""
        return prompt


# -------------------- AI Layer --------------------
class AIAssistant:
    def __init__(self, cfg: AppConfig) -> None:
        load_dotenv()
        self.gkey = os.getenv("GOOGLE_API_KEY")
        self.orkey = os.getenv("OPEN_ROUTER_API_KEY")
        self.use_gemini = cfg.use_gemini and bool(self.gkey) and genai is not None
        self.use_openrouter = cfg.use_openrouter and bool(self.orkey) and OpenAI is not None
        if self.use_gemini:
            try:
                genai.configure(api_key=self.gkey)
            except Exception as e:
                logger.warning("Không thể cấu hình Gemini: %s", e)
                self.use_gemini = False
        self._openrouter_client = None
        if self.use_openrouter:
            try:
                self._openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.orkey)
            except Exception as e:
                logger.warning("Không thể cấu hình OpenRouter: %s", e)
                self.use_openrouter = False

    def analyze_gemini(self, prompt: str) -> str:
        if not self.use_gemini:
            return "[Gemini] Bỏ qua (chưa cấu hình API hoặc tắt)"
        try:
            model = genai.GenerativeModel(model_name="gemini-2.5-flash")
            resp = model.generate_content(prompt)
            return resp.text.strip() if getattr(resp, "text", None) else "[Gemini] Không có phản hồi"
        except Exception as e:
            logger.error("Gemini error: %s", e)
            return f"[Gemini] Lỗi: {e}"

    def analyze_openrouter(self, prompt: str) -> str:
        if not self.use_openrouter:
            return "[OpenRouter] Bỏ qua (chưa cấu hình API hoặc tắt)"
        try:
            resp = self._openrouter_client.chat.completions.create(
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[{"role": "user", "content": prompt}],
            )
            if resp and resp.choices:
                return resp.choices[0].message.content
            return "[OpenRouter] Không có phản hồi"
        except Exception as e:
            logger.error("OpenRouter error: %s", e)
            return f"[OpenRouter] Lỗi: {e}"


# -------------------- Analyzer Orchestrator --------------------
class Analyzer:
    def __init__(self, cfg: AppConfig, fetcher: DataFetcher, weights: Weights) -> None:
        self.cfg = cfg
        self.fetcher = fetcher
        self.weights = weights
        self.ai = AIAssistant(cfg)

    def _build_history_text(self, symbol: str) -> str:
        path = os.path.join(self.cfg.data_dir, f"{symbol}_data.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path).tail(2000)
                return df.to_string(index=False, float_format="{:.2f}".format)
            except Exception as e:
                logger.warning("Không đọc được %s: %s", path, e)
        return "Không có dữ liệu lịch sử."

    def _build_info_text(self, symbol: str) -> str:
        path = os.path.join(self.cfg.data_dir, f"{symbol}_infor.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df.to_string(index=False, float_format="{:.2f}".format)
            except Exception as e:
                logger.warning("Không đọc được %s: %s", path, e)
        return "Không có dữ liệu thông tin công ty."

    def _build_market_text(self) -> str:
        path = os.path.join(self.cfg.data_dir, "market_filtered_pe.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df.to_string(index=False, float_format="{:.2f}".format)
            except Exception as e:
                logger.warning("Không đọc được %s: %s", path, e)
        return "Không có dữ liệu thông tin thị trường."

    def _tech_snapshot(self, sig: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rsi": sig.get("rsi_value"),
            "ma": {"ma10": sig.get("ma10"), "ma20": sig.get("ma20"), "ma50": sig.get("ma50"), "ma200": sig.get("ma200")},
            "bollinger_bands": {"upper": sig.get("bb_upper"), "lower": sig.get("bb_lower")},
            "macd": {"macd": sig.get("macd"), "signal": sig.get("macd_signal"), "histogram": sig.get("macd_hist")},
            "ichimoku": {"tenkan": sig.get("ichimoku_tenkan_sen"), "kijun": sig.get("ichimoku_kijun_sen"), "senkou_a": sig.get("ichimoku_senkou_span_a"), "senkou_b": sig.get("ichimoku_senkou_span_b"), "chikou": sig.get("ichimoku_chikou_span")},
            "volume": {"current": sig.get("volume"), "ma20": sig.get("volume_ma_20"), "ma50": sig.get("volume_ma_50")},
        }

    def analyze_one(self, symbol: str) -> Optional[Dict[str, Any]]:
        logger.info("%s", "=" * 60)
        logger.info("PHÂN TÍCH MÃ %s", symbol)
        logger.info("%s", "=" * 60)

        # Data
        df = self.fetcher.get_stock_data(symbol)
        if not validate_dataframe(df):
            logger.error("Thiếu dữ liệu %s", symbol)
            return None
        fin = self.fetcher.get_financial_data(symbol)
        info_text_full = self.fetcher.get_company_info(symbol)

        # Features
        df = FeatureEngineer.preprocess(df)
        if not validate_dataframe(df) or len(df) < 100:
            logger.error("Dữ liệu %s không đủ dài", symbol)
            return None
        df = FeatureEngineer.create_indicators(df)
        if symbol.upper() != self.cfg.vni_symbol:
            vni = self.fetcher.get_market_data()
            if validate_dataframe(vni):
                df = FeatureEngineer.relative_strength(df, vni)
            else:
                logger.warning("Không có VNI để tính RS")

        # Score
        scorer = Scorer(self.weights, self.cfg)
        score, signal = scorer.score(df, symbol)

        # Build prompt
        hist_text = self._build_history_text(symbol)
        info_text = self._build_info_text(symbol)
        market_text = self._build_market_text()
        tech = self._tech_snapshot(signal)
        if symbol.upper() == self.cfg.vni_symbol:
            prompt = PromptBuilder.vnindex_prompt(symbol, signal.get("current_price", 0.0), tech, hist_text, market_text)
        else:
            prompt = PromptBuilder.stock_prompt(symbol, signal.get("current_price", 0.0), tech, signal, fin, info_text_full, hist_text, info_text, market_text)

        # Persist prompt (thuận tiện debug)
        with open(os.path.join(self.cfg.data_dir, "prompt.txt"), "w", encoding="utf-8-sig") as f:
            f.write(prompt)

        # AI synthesis
        gemini_text = self.ai.analyze_gemini(prompt)
        # openrouter_text = self.ai.analyze_openrouter(prompt)
        print(gemini_text)
        # Log summary
        logger.info("Giá: %s | Tín hiệu: %s | Điểm: %.1f", f"{signal['current_price']:,}", signal["signal"], score)

        # Report
        report = {
            "symbol": symbol,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": signal.get("current_price"),
            "signal": signal.get("signal"),
            "recommendation": signal.get("recommendation"),
            "score": signal.get("score"),
            # copy essential metrics
            **{k: signal.get(k) for k in [
                "rsi_value", "ma10", "ma20", "ma50", "ma200", "rs", "rs_point", "open", "high", "low",
                "volume", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower", "volume_ma_20", "volume_ma_50",
                "ichimoku_tenkan_sen", "ichimoku_kijun_sen", "ichimoku_senkou_span_a", "ichimoku_senkou_span_b",
                "ichimoku_chikou_span", "rs_sma_10", "rs_sma_20", "rs_sma_50", "rs_sma_200",
                "rs_point_sma_10", "rs_point_sma_20", "rs_point_sma_50", "rs_point_sma_200",
                "relative_strength_3d", "relative_strength_1m", "relative_strength_3m", "relative_strength_1y"
            ]},
            "gemini_analysis": gemini_text,
            "openrouter_analysis": openrouter_text,
        }
        out_path = os.path.join(self.cfg.data_dir, f"{symbol}_report.json")
        with open(out_path, "w", encoding="utf-8-sig") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Đã lưu báo cáo: %s", out_path)
        return report

    def run_filter(self, min_market_cap: Optional[int] = None) -> None:
        self.fetcher.filter_stocks_low_pe_high_cap(min_market_cap)


# -------------------- CLI --------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vietnam Stock Analysis (VNStock + AI)")
    p.add_argument("tickers", nargs="*", help="Mã cổ phiếu, ví dụ: VCB FPT HPG hoặc VNINDEX")
    p.add_argument("--start", dest="start", default=AppConfig.default().start_date, help="Ngày bắt đầu YYYY-MM-DD")
    p.add_argument("--end", dest="end", default=AppConfig.default().end_date, help="Ngày kết thúc YYYY-MM-DD")
    p.add_argument("--min-cap", dest="mincap", type=int, default=500, help="Vốn hoá tối thiểu để lọc (tỷ VND)")
    p.add_argument("--no-gemini", action="store_true", help="Tắt phân tích Gemini")
    p.add_argument("--no-openrouter", action="store_true", help="Tắt phân tích OpenRouter")
    p.add_argument("--no-filter", action="store_true", help="Không chạy bước lọc cổ phiếu")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = AppConfig(
        start_date=args.start,
        end_date=args.end,
        min_market_cap=args.mincap,
        use_gemini=not args.no_gemini,
        use_openrouter=not args.no_openrouter,
    )
    fetcher = DataFetcher(cfg)
    analyzer = Analyzer(cfg, fetcher, Weights())
    if not args.no_filter:
        logger.info("🔍 Đang lọc cổ phiếu theo tiêu chí PE & vốn hoá...")
        analyzer.run_filter(cfg.min_market_cap)
    if not args.tickers:
        logger.info("Không có mã truyền vào. Ví dụ chạy: python vnstock_refactored.py VCB FPT")
        return
    for tk in (t.strip().upper() for t in args.tickers if t.strip()):
        try:
            analyzer.analyze_one(tk)
        except Exception as e:
            logger.error("Phân tích mã %s lỗi: %s", tk, e)
    logger.info("✅ Hoàn tất. Báo cáo trong thư mục %s", cfg.data_dir)


if __name__ == "__main__":
    main()
