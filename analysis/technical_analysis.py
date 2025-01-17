import pandas as pd
import numpy as np
from typing import Dict, Any
import pandas_ta as ta
from finta import TA

class TechnicalAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        # OHLCV formatına dönüştür
        self.data['open'] = self.data['price']
        self.data['high'] = self.data['price']
        self.data['low'] = self.data['price']
        self.data['close'] = self.data['price']
        
    def calculate_indicators(self) -> Dict[str, Any]:
        """
        Teknik göstergeleri hesaplar
        
        Returns:
            Dict[str, Any]: Hesaplanan göstergeler
        """
        indicators = {}
        
        # SMA (Simple Moving Average)
        indicators['sma_20'] = self._calculate_sma(20)
        indicators['sma_50'] = self._calculate_sma(50)
        
        # EMA (Exponential Moving Average)
        indicators['ema_12'] = self._calculate_ema(12)
        indicators['ema_26'] = self._calculate_ema(26)
        
        # RSI (Relative Strength Index)
        indicators['rsi'] = self._calculate_rsi()
        
        # MACD (Moving Average Convergence Divergence)
        macd_data = self._calculate_macd()
        indicators.update(macd_data)
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands()
        indicators.update(bb_data)
        
        return indicators
    
    def _calculate_sma(self, period: int) -> pd.Series:
        """Basit Hareketli Ortalama hesaplar"""
        return self.data.ta.sma(length=period)
    
    def _calculate_ema(self, period: int) -> pd.Series:
        """Üstel Hareketli Ortalama hesaplar"""
        return self.data.ta.ema(length=period)
    
    def _calculate_rsi(self, period: int = 14) -> pd.Series:
        """RSI (Göreceli Güç Endeksi) hesaplar"""
        return TA.RSI(self.data, period=period)
    
    def _calculate_macd(self) -> Dict[str, pd.Series]:
        """MACD (Hareketli Ortalama Yakınsama/Iraksama) hesaplar"""
        macd = self.data.ta.macd()
        return {
            'macd_line': macd['MACD_12_26_9'],
            'signal_line': macd['MACDs_12_26_9'],
            'macd_histogram': macd['MACDh_12_26_9']
        }
    
    def _calculate_bollinger_bands(self, period: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bantlarını hesaplar"""
        bb = self.data.ta.bbands(length=period, std=num_std)
        return {
            'bb_middle': bb['BBM_20_2.0'],
            'bb_upper': bb['BBU_20_2.0'],
            'bb_lower': bb['BBL_20_2.0']
        }
    
    def get_signals(self) -> Dict[str, str]:
        """
        Teknik göstergelere dayalı alım/satım sinyalleri üretir
        
        Returns:
            Dict[str, str]: Sinyal türleri ve değerleri
        """
        signals = {}
        indicators = self.calculate_indicators()
        
        # RSI Sinyalleri
        rsi = indicators['rsi'].iloc[-1]
        if rsi < 30:
            signals['rsi'] = 'Aşırı Satım'
        elif rsi > 70:
            signals['rsi'] = 'Aşırı Alım'
        else:
            signals['rsi'] = 'Nötr'
        
        # MACD Sinyalleri
        if indicators['macd_line'].iloc[-1] > indicators['signal_line'].iloc[-1]:
            signals['macd'] = 'Al'
        else:
            signals['macd'] = 'Sat'
        
        # Bollinger Bant Sinyalleri
        current_price = self.data['price'].iloc[-1]
        if current_price > indicators['bb_upper'].iloc[-1]:
            signals['bollinger'] = 'Aşırı Alım'
        elif current_price < indicators['bb_lower'].iloc[-1]:
            signals['bollinger'] = 'Aşırı Satım'
        else:
            signals['bollinger'] = 'Nötr'
        
        return signals 