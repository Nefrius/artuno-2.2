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
        self.lstm_model = None
        self.rf_model = None
        self.arima_model = None
        self.sequence_length = 24  # 24 saatlik veri
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Veriyi model için hazırlar"""
        # Veriyi ölçeklendir
        scaled_data = self.scaler.fit_transform(data[['price', 'volume']].values)
        
        # Sekans verisi oluştur
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length, 0])
            
        return np.array(X), np.array(y)
        
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> None:
        """LSTM modelini oluşturur"""
        self.lstm_model = LSTM(input_shape[1], 100, 2).to(self.device)
        self.optimizer = torch.optim.Adam(self.lstm_model.parameters())
        self.criterion = nn.MSELoss()
        
    def train_models(self, data: pd.DataFrame) -> None:
        """Tüm modelleri eğitir"""
        # Veriyi hazırla
        X, y = self.prepare_data(data)
        
        # LSTM modelini eğit
        if self.lstm_model is None:
            self.build_lstm_model((X.shape[1], X.shape[2]))
            
        dataset = TimeSeriesDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.lstm_model.train()
        for epoch in range(50):
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
        
        # Random Forest modelini eğit
        X_rf = X.reshape(X.shape[0], -1)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_rf, y)
        
        # ARIMA modelini eğit
        self.arima_model = ARIMA(data['price'], order=(5,1,0))
        self.arima_model = self.arima_model.fit()
        
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fiyat tahminlerini yapar"""
        # Son veriyi al
        last_sequence = self.scaler.transform(data[['price', 'volume']].values)[-self.sequence_length:]
        last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
        
        # LSTM tahmini
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_pred = self.lstm_model(last_sequence).cpu().numpy()[0][0]
        lstm_pred = self.scaler.inverse_transform(
            np.concatenate([lstm_pred.reshape(-1, 1), np.zeros((1, 1))], axis=1)
        )[0, 0]
        
        # Random Forest tahmini
        rf_pred = self.rf_model.predict(last_sequence.cpu().numpy().reshape(1, -1))[0]
        rf_pred = self.scaler.inverse_transform(
            np.concatenate([rf_pred.reshape(-1, 1), np.zeros((1, 1))], axis=1)
        )[0, 0]
        
        # ARIMA tahmini
        arima_forecast = self.arima_model.forecast(steps=24)
        
        # Saatlik tahminler
        hourly_predictions = self._generate_hourly_predictions(data, arima_forecast)
        
        # Piyasa durumu analizi
        market_state = self._analyze_market_state(data)
        
        # Güven skoru hesapla
        confidence_score = self._calculate_confidence(data, lstm_pred, rf_pred)
        
        return {
            'lstm_prediction': lstm_pred,
            'rf_prediction': rf_pred,
            'arima_predictions': arima_forecast,
            'hourly_predictions': hourly_predictions,
            'market_state': market_state,
            'confidence_score': confidence_score
        }
        
    def _generate_hourly_predictions(self, data: pd.DataFrame, arima_forecast: np.ndarray) -> List[Dict[str, Any]]:
        """Saatlik tahminler oluşturur"""
        predictions = []
        last_price = data['price'].iloc[-1]
        
        for i, price in enumerate(arima_forecast):
            confidence = max(0, 1 - (i * 0.02))  # Zaman ilerledikçe güven azalır
            change = ((price - last_price) / last_price) * 100
            
            predictions.append({
                'timestamp': datetime.now() + timedelta(hours=i+1),
                'predicted_price': price,
                'confidence': confidence,
                'change_percent': change
            })
            
        return predictions
        
    def _analyze_market_state(self, data: pd.DataFrame) -> Dict[str, str]:
        """Piyasa durumunu analiz eder"""
        df = data.copy()
        
        # RSI hesapla
        df.ta.rsi(close='price', length=14, append=True)
        rsi = df['RSI_14'].iloc[-1]
        
        # SMA hesapla
        df.ta.sma(close='price', length=20, append=True)
        df.ta.sma(close='price', length=50, append=True)
        sma20 = df['SMA_20'].iloc[-1]
        sma50 = df['SMA_50'].iloc[-1]
        
        # Risk seviyesi hesapla
        volatility = np.std(df['price'].values[-20:]) / np.mean(df['price'].values[-20:])
        
        return {
            'trend': 'Yükseliş Trendi' if sma20 > sma50 else 'Düşüş Trendi',
            'rsi_state': 'Aşırı Alım' if rsi > 70 else 'Aşırı Satım' if rsi < 30 else 'Normal',
            'risk_level': 'Yüksek' if volatility > 0.02 else 'Orta' if volatility > 0.01 else 'Düşük'
        }
        
    def _calculate_confidence(self, data: pd.DataFrame, lstm_pred: float, rf_pred: float) -> float:
        """Tahmin güven skorunu hesaplar"""
        # Model tahminleri arasındaki uyum
        pred_diff = abs(lstm_pred - rf_pred) / ((lstm_pred + rf_pred) / 2)
        model_agreement = max(0, 1 - pred_diff)
        
        # Veri kalitesi
        data_quality = min(1, len(data) / (self.sequence_length * 4))
        
        # Piyasa volatilitesi
        volatility = np.std(data['price'].values[-20:]) / np.mean(data['price'].values[-20:])
        volatility_score = max(0, 1 - (volatility * 50))
        
        # Ağırlıklı ortalama
        confidence = (0.4 * model_agreement + 0.3 * data_quality + 0.3 * volatility_score)
        
        return confidence 