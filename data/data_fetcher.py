import os
import requests
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any

class DataFetcher:
    def __init__(self):
        self.coingecko_api_key = os.getenv('NEXT_PUBLIC_COINGECKO_API_KEY')
        self.coinmarketcap_api_key = os.getenv('NEXT_PUBLIC_COINMARKETCAP_API_KEY')
        self.base_url_coingecko = 'https://api.coingecko.com/api/v3'
        self.base_url_coinmarketcap = 'https://pro-api.coinmarketcap.com/v1'
        
        # SQLite veritabanını başlat
        self.db_path = 'data/crypto.db'
        self._init_db()

    def _init_db(self):
        """Veritabanı ve tabloları oluşturur"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS price_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        coin TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        price REAL NOT NULL,
                        volume REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
        except Exception as e:
            print(f"Veritabanı başlatma hatası: {str(e)}")

    def fetch_data(self, coin: str, timeframe: str) -> pd.DataFrame:
        """
        Belirtilen kripto para birimi için veri toplar
        
        Args:
            coin (str): Kripto para birimi sembolü (örn. BTC)
            timeframe (str): Zaman aralığı (1h, 1d, 7d)
            
        Returns:
            pd.DataFrame: OHLCV verileri
        """
        try:
            # Zaman aralığını hesapla
            days = self._calculate_days(timeframe)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # CoinGecko'dan veri al
            params = {
                'vs_currency': 'usd',
                'from': int(start_time.timestamp()),
                'to': int(end_time.timestamp()),
                'x_cg_demo_api_key': self.coingecko_api_key
            }
            
            coin_id = self._get_coin_id(coin.lower())
            url = f"{self.base_url_coingecko}/coins/{coin_id}/market_chart/range"
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Verileri DataFrame'e dönüştür
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ek verileri ekle
            volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            volumes['timestamp'] = pd.to_datetime(volumes['timestamp'], unit='ms')
            volumes.set_index('timestamp', inplace=True)
            
            df['volume'] = volumes['volume']
            
            # Verileri veritabanına kaydet
            self._save_to_db(coin, df)
            
            return df
            
        except Exception as e:
            raise Exception(f"Veri toplama hatası: {str(e)}")
    
    def _save_to_db(self, coin: str, data: pd.DataFrame) -> None:
        """Verileri SQLite veritabanına kaydeder"""
        try:
            # DataFrame'i kayıtlara dönüştür
            records = data.reset_index()
            records['coin'] = coin
            
            # Veritabanına kaydet
            with sqlite3.connect(self.db_path) as conn:
                records.to_sql('price_history', conn, if_exists='append', index=False)
            
        except Exception as e:
            print(f"Veritabanı kayıt hatası: {str(e)}")
    
    def _calculate_days(self, timeframe: str) -> int:
        """Zaman aralığını gün cinsinden hesaplar"""
        timeframe_map = {
            '1h': 1,
            '1d': 1,
            '7d': 7,
            '30d': 30
        }
        return timeframe_map.get(timeframe, 7)
    
    def _get_coin_id(self, symbol: str) -> str:
        """Sembolden CoinGecko coin ID'sini alır"""
        try:
            url = f"{self.base_url_coingecko}/simple/supported_vs_currencies"
            response = requests.get(url)
            response.raise_for_status()
            
            # Basit eşleştirme için yaygın coinler
            common_coins = {
                'btc': 'bitcoin',
                'eth': 'ethereum',
                'bnb': 'binancecoin',
                'xrp': 'ripple',
                'doge': 'dogecoin',
                'ada': 'cardano',
                'sol': 'solana'
            }
            
            return common_coins.get(symbol, symbol)
            
        except Exception:
            return symbol  # Hata durumunda sembolü olduğu gibi döndür 