import os
from typing import Dict, Any
from dotenv import load_dotenv

class Config:
    def __init__(self):
        # .env dosyasını yükle
        load_dotenv()
        
        # API anahtarlarını al
        self.api_keys = {
            'supabase_url': os.getenv('NEXT_PUBLIC_SUPABASE_URL'),
            'supabase_anon_key': os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY'),
            'coingecko_api_key': os.getenv('NEXT_PUBLIC_COINGECKO_API_KEY'),
            'coinmarketcap_api_key': os.getenv('NEXT_PUBLIC_COINMARKETCAP_API_KEY'),
            'cryptopanic_api_key': os.getenv('CRYPTOPANIC_API_KEY'),
            'x_api_key': os.getenv('NEXT_PUBLIC_X_API_KEY'),
            'x_api_secret': os.getenv('NEXT_PUBLIC_X_API_SECRET')
        }
        
        # Varsayılan ayarlar
        self.settings = {
            'timeframes': ['1h', '1d', '7d', '30d'],
            'default_timeframe': '1d',
            'max_coins': 100,
            'cache_duration': 300,  # 5 dakika
            'update_interval': 60,  # 1 dakika
            'sentiment_weight': 0.1,  # Duygu analizi ağırlığı
            'technical_weight': 0.6,  # Teknik analiz ağırlığı
            'ml_weight': 0.3  # Makine öğrenimi ağırlığı
        }
    
    def get_api_key(self, service: str) -> str:
        """
        Belirtilen servis için API anahtarını döndürür
        
        Args:
            service (str): Servis adı
            
        Returns:
            str: API anahtarı
        """
        return self.api_keys.get(f'{service}_api_key', '')
    
    def get_setting(self, setting: str) -> Any:
        """
        Belirtilen ayarı döndürür
        
        Args:
            setting (str): Ayar adı
            
        Returns:
            Any: Ayar değeri
        """
        return self.settings.get(setting)
    
    def update_setting(self, setting: str, value: Any) -> None:
        """
        Belirtilen ayarı günceller
        
        Args:
            setting (str): Ayar adı
            value (Any): Yeni değer
        """
        if setting in self.settings:
            self.settings[setting] = value
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        API anahtarlarının geçerliliğini kontrol eder
        
        Returns:
            Dict[str, bool]: Her API anahtarının durumu
        """
        status = {}
        for service, key in self.api_keys.items():
            status[service] = bool(key)
        return status
    
    def get_all_settings(self) -> Dict[str, Any]:
        """
        Tüm ayarları döndürür
        
        Returns:
            Dict[str, Any]: Tüm ayarlar
        """
        return self.settings.copy()  # Güvenli kopya döndür 