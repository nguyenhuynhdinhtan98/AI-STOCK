from vnstock import Vnstock
## Bạn cũng có thể gọi các class trong theo vị trí chi tiết của nguồn dữ liệu
from vnstock.explorer.vci import Listing, Quote, Company, Finance, Trading

# hoặc

from vnstock.explorer.tcbs import Quote, Company, Finance, Trading, Screener
stock = Vnstock().stock(symbol='ACB', source='VCI')
stockvci = stock.listing.all_symbols()
stock.listing.symbols_by_exchange()
print(stockvci)