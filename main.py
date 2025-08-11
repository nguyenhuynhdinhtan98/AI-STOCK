import os
from vnstock import Vnstock
from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key="AIzaSyBh4yjR8V6ZNNUFsS-d_m3A9JWIKB__0n4")



stock = Vnstock().stock(symbol='ACB', source='VCI')
a = stock.quote.history(start='2025-01-01', end='2025-08-11', interval='1D')
print(a)


response = client.models.generate_content(
    model="gemini-2.5-flash", contents=f'Gia trung binh 10 ngay hien tai {a.to_string()}'
)
print(response.text)
