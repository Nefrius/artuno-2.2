import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class PricePredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.rf_model = None
        self.arima_model = None
        self.last_price = None
        
    def prepare_data(self, data):
        """Veriyi hazırlar ve teknik indikatörleri hesaplar"""
        try:
            # DataFrame kontrolü ve dönüşümü
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, dict):
                    df = pd.DataFrame(data)
                elif isinstance(data, pd.Series):
                    df = data.to_frame(name='price')
                else:
                    # Liste veya numpy array ise
                    if isinstance(data, (list, np.ndarray)):
                        index = pd.date_range(end=datetime.now(), periods=len(data), freq='H')
                        df = pd.DataFrame({'price': data}, index=index)
                    else:
                        # Tek bir değer ise
                        df = pd.DataFrame({'price': [float(data)]}, 
                                        index=[datetime.now()])
            else:
                df = data.copy()
            
            # Sütun isimlerini kontrol et
            if 'price' not in df.columns and len(df.columns) > 0:
                df = df.rename(columns={df.columns[0]: 'price'})
            
            # Veri tipini kontrol et
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            # NaN değerleri doldur
            df = df.ffill().bfill()
            
            # En az 2 satır veri olmalı
            if len(df) < 2:
                new_index = df.index[-1] + pd.Timedelta(hours=1)
                df.loc[new_index] = df.iloc[-1]
            
            # Teknik indikatörleri hesapla
            df['RSI_14'] = df['price'].rolling(window=14).apply(
                lambda x: 100 - (100 / (1 + (x[x > 0].mean() / -x[x < 0].mean())))
                if len(x[x > 0]) > 0 and len(x[x < 0]) > 0 
                else 50
            ) if len(df) >= 14 else pd.Series([50] * len(df), index=df.index)
            
            df['sma20'] = df['price'].rolling(window=20).mean()
            df['sma50'] = df['price'].rolling(window=50).mean()
            
            # NaN değerleri doldur
            df = df.ffill().bfill()
            
            self.last_price = float(df['price'].iloc[-1])
            
            # Hacim verisi yoksa ekle
            if 'volume' not in df.columns:
                df['volume'] = 0
            
            # Eksik teknik indikatörleri doldur
            df['RSI_14'] = df['RSI_14'].fillna(50)
            df['sma20'] = df['sma20'].fillna(df['price'])
            df['sma50'] = df['sma50'].fillna(df['price'])
            
            features = ['price', 'volume', 'RSI_14', 'sma20', 'sma50']
            normalized_data = self.scaler.fit_transform(df[features])
            
            X = normalized_data[:-1]
            y = normalized_data[1:, 0]
            
            self.X_test = normalized_data[-1:].reshape(1, -1)
            
            return X, y
            
        except Exception as e:
            print(f"Veri hazırlama hatası: {str(e)}")
            raise ValueError(f"Veri hazırlanamadı: {str(e)}")
        
    def train_models(self, data):
        """Modelleri eğit"""
        X, y = self.prepare_data(data)
        
        # Random Forest modelini eğit
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)
        
        # ARIMA modelini eğit
        try:
            price_series = pd.Series(
                data['price'].values,
                index=pd.date_range(start='2023-01-01', periods=len(data), freq='h')
            )
            self.arima_model = ARIMA(price_series, order=(2,1,2))
            self.arima_model = self.arima_model.fit()
        except Exception as e:
            print(f"ARIMA model hatası: {str(e)}")
            self.arima_model = None
        
        return True
        
    def predict(self, timeframe='1h', data=None):
        """
        Fiyat tahminleri yapar.
        
        Args:
            timeframe (str): Tahmin zaman aralığı ('1h', '1d', '7d', '30d')
            data (pd.DataFrame, optional): Güncel veri. Eğer verilirse, modeli güncellemek için kullanılır.
        """
        if self.X_test is None:
            raise ValueError("Model henüz eğitilmedi!")
            
        # Eğer yeni veri verildiyse, son durumu güncelle
        if data is not None:
            _, _ = self.prepare_data(data)
            
        predictions = []
        current_time = datetime.now()
        # Dakikayı 10'un katına yuvarla
        rounded_minutes = (current_time.minute // 10) * 10
        current_time = current_time.replace(minute=rounded_minutes, second=0, microsecond=0)
        
        if timeframe == '1h':
            num_predictions = 6  # 10'ar dakikalık 6 tahmin
        else:
            num_predictions = 24  # Saatlik 24 tahmin
            
        base_price = self.last_price
        volatility = 0.005  # %0.5 volatilite
        
        # RF tahminini al
        rf_pred = self.rf_model.predict(self.X_test)[0]
        rf_pred = self.scaler.inverse_transform(
            np.concatenate([rf_pred.reshape(-1, 1), np.zeros((1, self.X_test.shape[1]-1))], axis=1)
        )[0, 0]
        
        for i in range(num_predictions):
            if timeframe == '1h':
                prediction_time = current_time + timedelta(minutes=(i+1)*10)  # Gelecek 10'ar dakika
            else:
                prediction_time = current_time + timedelta(hours=i+1)  # Gelecek saatler
                
            # Tahmin için RF ve ARIMA'yı birleştir
            if self.arima_model:
                arima_pred = self.arima_model.forecast(1)[0]
                predicted_price = (rf_pred + arima_pred) / 2
            else:
                predicted_price = rf_pred
            
            # Rastgele değişim ekle
            price_change = np.random.normal(0, volatility)
            predicted_price *= (1 + price_change)
            
            confidence = max(0.95 - (i * 0.05), 0.5)
            
            price_range = (
                predicted_price * (1 - volatility * (1 - confidence)),
                predicted_price * (1 + volatility * (1 - confidence))
            )
            
            predictions.append({
                'timestamp': prediction_time,
                'price': predicted_price,
                'confidence': confidence,
                'change': price_change * 100,
                'range': price_range
            })
            
            base_price = predicted_price
            
        return predictions 