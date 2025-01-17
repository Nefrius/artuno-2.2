# Artuno - Kripto Para Tahmin Aracı

Artuno, Python tabanlı yerel bir kripto para tahmin aracıdır. Detaylı veri analizi, grafik analizi ve seçilen kripto paralar için doğru tahminler sağlar.

## Özellikler

- Gerçek zamanlı ve geçmiş kripto para verilerinin analizi
- Teknik göstergeler ve grafik analizleri
- Duygu analizi ve haber takibi
- Makine öğrenimi tabanlı fiyat tahminleri
- Terminal tabanlı kullanıcı arayüzü

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Çevre değişkenlerini ayarlayın:
- `.env` dosyası oluşturun ve API anahtarlarınızı ekleyin:
```
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_key
NEXT_PUBLIC_COINGECKO_API_KEY=your_coingecko_key
NEXT_PUBLIC_COINMARKETCAP_API_KEY=your_coinmarketcap_key
CRYPTOPANIC_API_KEY=your_cryptopanic_key
```

## Kullanım

```bash
python main.py
```

## Proje Yapısı

```
artuno/
├── data/               # Veri işleme modülleri
├── analysis/          # Analiz araçları
├── ml/                # Makine öğrenimi modelleri
├── utils/             # Yardımcı fonksiyonlar
├── visualization/     # Görselleştirme araçları
└── tests/             # Test dosyaları
```

## Lisans

MIT License 