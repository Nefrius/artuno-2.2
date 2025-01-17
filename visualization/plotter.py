import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from typing import Dict, Any, List
from datetime import datetime, timedelta

class DataVisualizer:
    def __init__(self):
        plt.style.use('seaborn-darkgrid')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100
        
    def plot_predictions(self, data: pd.DataFrame, predictions: Dict[str, Any]) -> None:
        """Tahminleri grafiğe çizer"""
        plt.figure(figsize=(15, 10))
        
        # Gerçek veriler
        plt.plot(data.index[-48:], data['price'].iloc[-48:], 
                 label='Gerçek Fiyat', color='blue', linewidth=2)
        
        # Son tarih
        last_date = data.index[-1]
        
        # Gelecek 24 saat için tahminler
        future_dates = pd.date_range(start=last_date, periods=25, freq='h')[1:]
        hourly_predictions = predictions['hourly_predictions']
        
        # Tahmin çizgisi
        predicted_prices = [p['predicted_price'] for p in hourly_predictions]
        plt.plot(future_dates, predicted_prices, 
                 label='Tahmin', color='red', linestyle='--', linewidth=2)
        
        # Tahmin aralıkları
        confidence_intervals = np.array([p['confidence'] for p in hourly_predictions])
        upper_bound = predicted_prices * (1 + 0.02 * confidence_intervals)
        lower_bound = predicted_prices * (1 - 0.02 * confidence_intervals)
        plt.fill_between(future_dates, lower_bound, upper_bound, 
                        color='red', alpha=0.1, label='Tahmin Aralığı')
        
        # Önemli saatlerdeki tahminleri işaretle
        for i, pred in enumerate(hourly_predictions):
            if i % 4 == 0:  # Her 4 saatte bir
                price = pred['predicted_price']
                timestamp = pred['timestamp']
                change = pred['change_percent']
                plt.scatter(timestamp, price, color='darkred', zorder=5)
                plt.annotate(f'{timestamp.strftime("%H:%M")}\n${price:.2f}\n{change:+.1f}%',
                            (timestamp, price),
                            xytext=(10, 10),
                            textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->'))
        
        # Grafik düzeni
        plt.title('Fiyat Tahmini (24 Saat)', fontsize=14, pad=20)
        plt.xlabel('Tarih', fontsize=12)
        plt.ylabel('Fiyat ($)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left')
        
        # Tarih formatını ayarla
        plt.gcf().autofmt_xdate()
        
        # Göster
        plt.tight_layout()
        plt.show()

    def plot_advanced_analysis(self, data: pd.DataFrame, predictions: Dict[str, Any]) -> None:
        """Gelişmiş analiz grafiğini çizer"""
        plt.style.use('seaborn')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        fig.suptitle('Fiyat ve Hacim Analizi', fontsize=14)
        
        # Fiyat grafiği
        ax1.plot(data.index, data['price'], label='Gerçek Fiyat', color='blue')
        
        # Tahmin grafiği
        pred_times = [p['timestamp'] for p in predictions]
        pred_prices = [p['price'] for p in predictions]
        pred_ranges = [p['range'] for p in predictions]
        
        ax1.plot(pred_times, pred_prices, 'r--', label='Tahmin', linewidth=2)
        
        # Tahmin aralığı
        range_low = [r[0] for r in pred_ranges]
        range_high = [r[1] for r in pred_ranges]
        ax1.fill_between(pred_times, range_low, range_high, color='red', alpha=0.1)
        
        ax1.set_title('Fiyat Grafiği')
        ax1.set_xlabel('Zaman')
        ax1.set_ylabel('Fiyat ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Hacim grafiği
        ax2.bar(data.index, data['volume'], color='gray', alpha=0.5, label='Hacim')
        ax2.set_title('Hacim Grafiği')
        ax2.set_xlabel('Zaman')
        ax2.set_ylabel('Hacim')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def print_predictions_table(self, predictions):
        """10 dakikalık tahminleri tablo formatında yazdırır"""
        print("\n10 Dakikalık Tahminler:")
        print("=" * 50)
        print(f"{'Saat':^10} | {'Fiyat':^12} | {'Değişim':^10} | {'Güven':^8}")
        print("-" * 50)
        
        for pred in predictions:
            time_str = pred['timestamp'].strftime('%H:%M')
            price = f"${pred['price']:.3f}"
            change = f"{pred['change']:+.2f}%"
            confidence = f"{pred['confidence']*100:.0f}%"
            
            print(f"{time_str:^10} | {price:^12} | {change:^10} | {confidence:^8}") 