# TV Dizisi Sohbet Botu (RAG Tabanlı)

## 📋 Genel Bakış

Bu proje, TV dizilerinin altyazı dosyalarından öğrenip, kullanıcıların diziler hakkında sorular sormasını sağlayan gelişmiş bir RAG (Retrieval Augmented Generation) sistemidir. Stranger Things ve Breaking Bad dizileri için çalışır durumda olup, çoklu dizi sorgulamayı destekler.

## 🎯 Temel Özellikler

- **Akıllı Sorgu Optimizasyonu**: Türkçe soruları İngilizce'ye çevirerek arama terimlerini genişletir
- **Otomatik Dizi Tespiti**: Soru içeriğinden otomatik olarak hangi dizi ile ilgili olduğunu anlar
- **Sezon/Bölüm Filtreleme**: Belirli sezon veya bölümlerden sorgulama yapar
- **Çoklu Dizi Desteği**: Tek seferde birden fazla dizi üzerinde arama yapabilir
- **Hibrit LLM Desteği**: Hem yerel Ollama (qwen2.5:7b) hem de Google Gemini API destekli
- **FastAPI REST API**: Web uygulamaları için hazır API endpoint'leri

## 📁 Klasör Yapısı

```
balcim_chatbot_projesi_v1.2/
│
├── api.py                          # FastAPI REST API servisi
├── main.py                         # CLI ile veri işleme aracı
├── index.html                      # Web arayüzü (opsiyonel)
├── requirements.txt                # Python bağımlılıkları
│
├── config/                         # Yapılandırma dosyaları
│   ├── constants.py                # LLM, embedding ve sistem sabitleri
│   └── paths.py                    # Dosya yolu yönetimi
│
├── data/                           # Veri deposu
│   ├── raw/                        # Ham altyazı dosyaları (.srt, .xlsx)
│   │   ├── breaking_bad/
│   │   │   ├── audio_descriptions/
│   │   │   └── captioned_subtitles/
│   │   └── stranger_things/
│   │       ├── audio_descriptions/
│   │       └── captioned_subtitles/
│   │
│   ├── processed/                  # İşlenmiş JSON dosyaları
│   │   ├── breaking_bad/
│   │   │   ├── captioned_subtitles/    # SRT'den dönüştürülmüş
│   │   │   ├── audio_descriptions/     # Excel'den dönüştürülmüş
│   │   │   └── merged/                 # Birleştirilmiş diyalog+aksiyon
│   │   └── stranger_things/
│   │       └── (aynı yapı)
│   │
│   └── chroma_db/                  # Vektör veritabanı
│       ├── breaking_bad/
│       └── stranger_things/
│
└── src/                            # Kaynak kod
    ├── vector_store.py             # Vektör DB yönetimi
    │
    ├── core/                       # Ana sistem bileşenleri
    │   ├── data_processor.py       # Ham veriyi işleme pipeline
    │   ├── llm_engine.py           # LLM yönetimi (Ollama/Gemini)
    │   ├── multi_series_service.py # Çoklu dizi yönetimi
    │   └── pipeline.py             # RAG chain oluşturma
    │
    ├── models/                     # Veri modelleri
    │   └── scene.py                # Sahne yapıları
    │
    ├── preprocessing/              # Veri ön işleme
    │   ├── srt_parser.py           # SRT altyazı parser
    │   ├── excel_parser.py         # Excel aksiyon betimlemeleri parser
    │   └── merger.py               # Diyalog + aksiyon birleştirme
    │
    ├── prompts/                    # LLM prompt şablonları
    │   ├── answer_prompt.py        # Cevap üretme prompt'u
    │   └── rewrite_prompt.py       # Sorgu optimizasyonu prompt'u
    │
    └── utils/                      # Yardımcı araçlar
        ├── data_loader.py          # JSON veri yükleme
        ├── logging.py              # Loglama yapılandırması
        ├── text_processing.py      # Metin işleme fonksiyonları
        └── validators.py           # Girdi doğrulama
```

## 🔄 Sistem Akış Mantığı

### 1. Veri İşleme Pipeline (main.py)

```
Ham Altyazı (.srt) → SRT Parser → JSON (diyalog)
                                    ↓
Ham Betimlemeler (.xlsx) → Excel Parser → JSON (aksiyon)
                                    ↓
                            Merger → Birleşik JSON
                                    ↓
                            Text Splitter (1000 char chunks)
                                    ↓
                    Google Embedding API (text-embedding-004)
                                    ↓
                            ChromaDB Vektör Store
```

**Önemli Detaylar:**
- **SRT Parser**: Zaman damgaları, konuşma metinleri ve metadata (sezon/bölüm) çıkarır
- **Excel Parser**: Audio description dosyalarındaki görsel betimlemeleri filtreler
- **Merger**: Aynı zaman aralığındaki diyalog ve aksiyonları n-gram ve zaman penceresi ile eşleştirir
- **Chunk Stratejisi**: 1000 karakter chunks, 150 karakter overlap (bağlam sürekliliği için)

### 2. Sorgu Pipeline (api.py)

```
Kullanıcı Sorusu (Türkçe)
        ↓
Query Optimizer (LLM)
├─ Türkçe → İngilizce çeviri
├─ Anahtar kelime genişletme (30-50 terim)
├─ Sezon/bölüm filtreleri çıkarma
└─ Otomatik dizi tespiti
        ↓
Filtered Vector Search (ChromaDB)
├─ Similarity search (k=5)
└─ Metadata filtering (season/episode)
        ↓
Retrieved Context (5 en ilgili chunk)
        ↓
Answer Generation (LLM)
└─ Context + Optimized Query → Türkçe Cevap
        ↓
API Response (JSON)
├─ answer: Türkçe cevap
├─ sources: Kaynak metadata
├─ optimized_query: İşlenmiş sorgu
└─ series_queried: Sorgulanan diziler
```

## 🤖 Model Seçimleri ve Gerekçeleri

### Embedding Model: Google text-embedding-004

**Neden?**
- **Çok Dilli Destek**: Türkçe ve İngilizce karışık sorguları iyi anlar
- **Uzun Bağlam**: 2048 token'a kadar metinleri embed edebilir
- **Semantik Anlama**: Karakter isimleri, duygu ve bağlam ilişkilerini güçlü yakalar
- **API Tabanlı**: Yerel kaynak gerektirmez, tutarlı performans

**Alternatifler:**
- sentence-transformers (yerel, ama Türkçe performansı düşük)
- OpenAI embeddings (daha pahalı, benzer performans)

### LLM: Dual Mode (Qwen2.5:7b + Gemini-1.5-Flash)

#### Qwen2.5:7b (Varsayılan - Yerel Ollama)

**Neden?**
- **Gizlilik**: Veriler yerel kalır
- **Maliyet**: API ücretleri yok
- **Hız**: Küçük model, düşük latency
- **Türkçe**: 7B parametreye rağmen iyi Türkçe üretir

**Kullanım Alanları:**
- Geliştirme ve test
- Hassas veri içeren sorgular
- Yüksek hacimli API çağrıları

#### Google Gemini-3-Flash-Preview (Opsiyonel)

**Neden?**
- **Kalite**: Daha karmaşık muhakeme
- **Uzun Bağlam**: 1M token context window
- **API Güvenilirliği**: Google altyapısı
- **Hız**: Flash versiyonu optimize edilmiş

**Kullanım Alanları:**
- Üretim ortamı (yüksek kalite gerekli)
- Karmaşık çıkarım gerektiren sorular
- Çoklu dizi karşılaştırma

### Vektör DB: ChromaDB

**Neden?**
- **Basit Entegrasyon**: Python-first, minimal setup
- **Metadata Filtering**: Sezon/bölüm gibi yapısal filtreler
- **Persistent Storage**: Disk tabanlı, yeniden embed gerektirmez
- **Performans**: Küçük-orta ölçekli projeler için yeterli (< 1M chunks)

## ⚙️ Kurulum

### 1. Gereksinimler

```bash
Python 3.11+
Ollama (yerel LLM için, opsiyonel)
```

### 2. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 3. Ortam Değişkenleri

`.env` dosyası oluşturun:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Veri İşleme

Ham altyazı dosyalarını `data/raw/` klasörüne yerleştirin, ardından:

```bash
python main.py --process --series stranger_things
python main.py --process --series breaking_bad
```

### 5. API'yi Başlat

```bash
uvicorn api:app --reload --port 8000
```

## 🔍 Kullanım Örnekleri

### API Üzerinden

```bash
# Tek dizi sorgusu
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Hopper öldü mü?", "series": "stranger_things"}'

# Sezon filtrelemeli
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Will nerede kayboldu?", "series": "stranger_things", "season": 1}'

# Tüm dizilerde ara
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Ana karakter kim?", "series": "all"}'
```

### CLI Üzerinden

```bash
# Veri işleme
python main.py --process --series breaking_bad
```

## 🛠️ Yapılandırma

[config/constants.py](config/constants.py) dosyasında özelleştirilebilir parametreler:

- `CHUNK_SIZE`: 1000 (chunk boyutu)
- `RETRIEVAL_K`: 5 (döndürülecek belge sayısı)
- `LLM_TEMPERATURE`: 0.2 (yaratıcılık seviyesi)
- `USE_LOCAL_LLM`: True (Ollama kullan/kullanma)
- `LOCAL_MODEL_NAME`: "qwen2.5:7b"
- `GOOGLE_MODEL_NAME`: "gemini-3-flash-preview"

## 🚀 Geliştirme Önerileri

1. **Daha Fazla Dizi**: Yeni diziler için `data/raw/<series_name>/` klasörüne veri ekle
2. **Gelişmiş Filtreleme**: Karakter isimleri, lokasyonlar için metadata ekle
3. **Konuşma Geçmişi**: Çok turlu diyalog desteği
4. **Web UI**: index.html dosyasını geliştirerek kullanıcı dostu arayüz
5. **Önbellekleme**: Sık sorulan sorular için Redis cache

## 📝 Lisans

Bu proje MAT409 dersi kapsamında geliştirilmiştir.

## 👤 Geliştirici

Emin - MAT409 Proje Çalışması
