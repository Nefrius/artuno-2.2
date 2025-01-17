#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import time
from ml.predictor import PricePredictor
from visualization.plotter import DataVisualizer
from data.data_fetcher import DataFetcher

SUPPORTED_COINS = {
    'BTC': 'Bitcoin',
    'ETH': 'Ethereum',
    'BNB': 'Binance Coin',
    'XRP': 'Ripple',
    'DOGE': 'Dogecoin',
    'ADA': 'Cardano',
    'SOL': 'Solana',
    'DOT': 'Polkadot',
    'MATIC': 'Polygon',
    'AVAX': 'Avalanche',
    'LINK': 'Chainlink',
    'UNI': 'Uniswap',
    'ATOM': 'Cosmos'
}

TIMEFRAMES = {
    '1h': '1 Saatlik',
    '1d': '1 Günlük',
    '7d': '7 Günlük',
    '30d': '30 Günlük'
}

def print_menu():
    """Ana menüyü yazdırır"""
    print("\n" + "=" * 50)
    print("Artuno - Kripto Para Tahmin Aracı")
    print("=" * 50)
    print("\nDesteklenen Coinler:")
    
    # Coinleri 3 sütun halinde yazdır
    coins = list(SUPPORTED_COINS.items())
    for i in range(0, len(coins), 3):
        row = coins[i:i+3]
        print("  ".join(f"{coin}: {name:<15}" for coin, name in row))

def print_timeframes():
    """Zaman aralıklarını yazdırır"""
    print("\nZaman Aralıkları:")
    for code, desc in TIMEFRAMES.items():
        print(f"{code}: {desc}")

def print_analysis_report(coin: str, data: pd.DataFrame, predictions: list):
    """Detaylı analiz raporu yazdırır"""
    print("\n" + "=" * 50)
    print(f"{SUPPORTED_COINS.get(coin, coin)} Analiz Raporu")
    print("=" * 50)
    
    # Fiyat hassasiyetini belirle
    current_price = data['price'].iloc[-1]
    if current_price < 0.1:
        price_format = ".6f"
    elif current_price < 1:
        price_format = ".4f"
    elif current_price < 100:
        price_format = ".3f"
    else:
        price_format = ".2f"
    
    # Fiyat bilgileri
    print("\nFiyat Analizi:")
    print(f"Mevcut Fiyat: ${current_price:{price_format}}")
    
    # Değişim oranları
    changes = {
        '1s': data['price'].pct_change().iloc[-1],
        '1h': data['price'].iloc[-1] / data['price'].iloc[-1] - 1,
        '24s': data['price'].iloc[-1] / data['price'].iloc[-24] - 1,
        '7g': data['price'].iloc[-1] / data['price'].iloc[-168] - 1 if len(data) > 168 else None
    }
    
    print("\nDeğişim Oranları:")
    print(f"Son 1 saat: {changes['1s']*100:+.2f}%")
    print(f"Son 24 saat: {changes['24s']*100:+.2f}%")
    if changes['7g'] is not None:
        print(f"Son 7 gün: {changes['7g']*100:+.2f}%")
    
    print(f"\nFiyat Aralığı (24s):")
    print(f"En Yüksek: ${data['price'].iloc[-24:].max():{price_format}}")
    print(f"En Düşük: ${data['price'].iloc[-24:].min():{price_format}}")
    print(f"Ortalama: ${data['price'].iloc[-24:].mean():{price_format}}")
    
    # Hacim analizi
    print("\nHacim Analizi:")
    volume_change = (data['volume'].iloc[-1] / data['volume'].iloc[-24] - 1) * 100
    print(f"24s Hacim: {data['volume'].iloc[-1]:,.0f}")
    print(f"Hacim Değişimi: {volume_change:+.2f}%")
    
    # Tahminler
    print("\nTahminler:")
    for pred in predictions:
        timestamp = pred['timestamp'].strftime('%H:%M')
        price = pred['price']
        confidence = pred['confidence'] * 100
        change = pred['change']
        range_low, range_high = pred['range']
        
        print(f"\nSaat {timestamp}")
        print(f"Tahmin Edilen Fiyat: ${price:{price_format}}")
        print(f"Değişim: {change:+.2f}%")
        print(f"Güven: {confidence:.1f}%")
        print(f"Aralık: ${range_low:{price_format}} - ${range_high:{price_format}}")
        
        # Güven göstergesi
        if confidence >= 90:
            print("⭐⭐⭐⭐⭐")
        elif confidence >= 80:
            print("⭐⭐⭐⭐")
        elif confidence >= 70:
            print("⭐⭐⭐")
        elif confidence >= 60:
            print("⭐⭐")
        else:
            print("⭐")

def main():
    fetcher = DataFetcher()
    visualizer = DataVisualizer()
    
    while True:
        try:
            print_menu()
            
            # Coin seçimi
            coin = input("\nKripto para birimi girin (örn. BTC) veya çıkmak için 'q': ").upper()
            if coin == 'Q':
                print("\nProgram sonlandırılıyor...")
                break
            
            if coin not in SUPPORTED_COINS:
                print(f"\nHata: {coin} desteklenmiyor. Lütfen listeden bir coin seçin.")
                continue
            
            # Zaman aralığı seçimi
            print_timeframes()
            timeframe = input("Zaman aralığı seçin: ")
            if timeframe not in TIMEFRAMES:
                print(f"\nHata: Geçersiz zaman aralığı. Lütfen listeden seçim yapın.")
                continue
            
            # Otomatik güncelleme seçeneği
            auto_update = input("\nOtomatik güncelleme istiyor musunuz? (e/h): ").lower() == 'e'
            update_interval = 300  # 5 dakika
            
            while True:
                # Veri çekme ve analiz
                print(f"\n{SUPPORTED_COINS[coin]} verileri getiriliyor...")
                data = fetcher.fetch_data(coin, timeframe)
                
                print("Model eğitiliyor...")
                predictor = PricePredictor()
                predictor.train_models(data)
                
                print("Tahminler yapılıyor...")
                predictions = predictor.predict(timeframe=timeframe)
                
                # Analiz raporu
                print_analysis_report(coin, data, predictions)
                
                # 10 dakikalık tahminler
                visualizer.print_predictions_table(predictions)
                
                # Grafikleri göster
                print("\nGrafikler hazırlanıyor...")
                visualizer.plot_advanced_analysis(data, predictions)
                
                if not auto_update:
                    break
                    
                print(f"\nBir sonraki güncelleme {update_interval} saniye sonra...")
                time.sleep(update_interval)
            
        except KeyboardInterrupt:
            print("\nProgram kullanıcı tarafından durduruldu.")
            break
            
        except Exception as e:
            print(f"\nHata oluştu: {str(e)}")
            print("Lütfen tekrar deneyin.")
            continue

if __name__ == '__main__':
    main() 