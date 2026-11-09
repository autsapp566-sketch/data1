import pandas as pd
import requests
import numpy as np

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_advanced_crypto_data(symbol="BTCUSDT", interval="1h", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"فشل في جلب بيانات {symbol}")
        return None
        
    data = response.json()
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    # اختيار الأعمدة الأساسية وتحويل الأنواع
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    # إضافة ميزات قيمة (Feature Engineering)
    df['price_change'] = df['close'].diff()
    df['price_change_percent'] = df['close'].pct_change() * 100
    df['MA_20'] = df['close'].rolling(window=20).mean() # المتوسط المتحرك 20
    df['MA_50'] = df['close'].rolling(window=50).mean() # المتوسط المتحرك 50
    df['RSI'] = calculate_rsi(df['close'], period=14)    # مؤشر القوة النسبية
    
    # إضافة عمود يحدد اسم العملة
    df['symbol'] = symbol
    
    return df

# قائمة العملات المستهدفة
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
all_dataframes = []

print("جاري سحب وتجهيز البيانات من Binance...")
for sym in symbols:
    df_temp = fetch_advanced_crypto_data(symbol=sym, interval="1h", limit=1000)
    if df_temp is not None:
        all_dataframes.append(df_temp)

# دمج كل البيانات في جدول واحد ضخم (Combined Dataset)
if all_dataframes:
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    # تنظيف الصفوف الفارغة الناتجة عن المؤشرات التقنية
    master_df.dropna(inplace=True)
    
    # حفظ الملفات بصيغ مختلفة
    master_df.to_parquet("crypto_market_historical_1h.parquet", index=False)
    # حفظ بصيغة CSV باستعمال الفاصلة المنقوطة (;) ليتوافق تماماً مع الإكسل
    master_df.to_csv("crypto_market_historical_1h.csv", index=False, sep=';')
    
    print("تم الانتهاء بنجاح! تم حفظ الملفات:")
    print("- crypto_market_historical_1h.parquet")
    print("- crypto_market_historical_1h.csv")
else:
    print("حدث خطأ أثناء جلب البيانات.")