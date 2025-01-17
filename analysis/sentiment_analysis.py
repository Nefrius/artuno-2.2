import os
import requests
from typing import Dict, Any, List
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    def __init__(self):
        self.cryptopanic_api_key = os.getenv('CRYPTOPANIC_API_KEY')
        self.base_url = 'https://cryptopanic.com/api/v1'
        
        # NLTK'yı hazırla
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze(self, coin: str) -> float:
        """
        Belirtilen kripto para birimi için duygu analizi yapar
        
        Args:
            coin (str): Kripto para birimi sembolü
            
        Returns:
            float: -1 ile 1 arasında bir duygu skoru
        """
        try:
            # Haberleri getir
            news_data = self._fetch_news(coin)
            
            if not news_data:
                return 0.0
            
            # Her haber için duygu analizi yap
            sentiment_scores = []
            for news in news_data:
                title = news.get('title', '')
                score = self.sia.polarity_scores(title)
                sentiment_scores.append(score['compound'])
            
            # Ortalama duygu skorunu hesapla
            if sentiment_scores:
                return sum(sentiment_scores) / len(sentiment_scores)
            return 0.0
            
        except Exception as e:
            print(f"Duygu analizi hatası: {str(e)}")
            return 0.0
    
    def _fetch_news(self, coin: str) -> List[Dict[str, Any]]:
        """CryptoPanic API'den haber verileri getirir"""
        try:
            params = {
                'auth_token': self.cryptopanic_api_key,
                'currencies': coin,
                'kind': 'news',
                'filter': 'important',
                'public': 'true'
            }
            
            response = requests.get(f"{self.base_url}/posts/", params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get('results', [])
            
        except Exception as e:
            print(f"Haber getirme hatası: {str(e)}")
            return []
    
    def get_sentiment_summary(self, coin: str) -> Dict[str, Any]:
        """
        Duygu analizi özeti oluşturur
        
        Args:
            coin (str): Kripto para birimi sembolü
            
        Returns:
            Dict[str, Any]: Duygu analizi özeti
        """
        sentiment_score = self.analyze(coin)
        
        # Duygu durumunu belirle
        if sentiment_score >= 0.5:
            sentiment = "Çok Olumlu"
        elif sentiment_score >= 0.1:
            sentiment = "Olumlu"
        elif sentiment_score >= -0.1:
            sentiment = "Nötr"
        elif sentiment_score >= -0.5:
            sentiment = "Olumsuz"
        else:
            sentiment = "Çok Olumsuz"
        
        return {
            'score': sentiment_score,
            'sentiment': sentiment,
            'timestamp': datetime.now().isoformat()
        } 