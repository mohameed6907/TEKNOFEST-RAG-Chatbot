# TEKNOFEST RAG Chatbot

FastAPI, LangChain ve LangGraph tabanli, TEKNOFEST odakli RAG destekli sohbet asistani.

## Özellikler

- FastAPI ile tek sayfalık chatbot arayüzü
- Yerel RAG klasörü (`RAG/raw`) üzerinden PDF/DOCX/TXT/MD dokümanları indeksleme
- `https://teknofest.org/tr/` sitesini 2 derinliğe kadar crawl edip vektör indeks oluşturma
- Soruya göre çok kademeli yanıt stratejisi:
  - Genel sorular → LLM doğrudan cevap
  - TEKNOFEST soruları → önce yerel dokümanlar, sonra TEKNOFEST site indeksleri, en son Tavily web araması
- Halusinasyon denetimi icin ek self-check adimi
- Groq birincil LLM, DeepSeek/Kimi/OpenAI icin provider bazli gecis mimarisi
- OpenAI tabanlı embedding (text-embedding-3-small)

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate  # Windows için: .venv\\Scripts\\activate
pip install -r requirements.txt
```

`.env` dosyanızı oluşturun:

```bash
cp .env.example .env  # Linux/macOS
```

ve asagidaki degiskenleri doldurun:

```bash
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-70b-versatile
GROQ_API_KEY=...
DEEPSEEK_API_KEY=
KIMI_API_KEY=
TAVILY_API_KEY=...
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL_NAME=text-embedding-3-small
RAG_CONFIDENCE_THRESHOLD=0.55
```

## RAG Klasörü ve İndeksleme

- Yerel dokümanlarınızı proje kökündeki `RAG/raw/` klasörüne kopyalayın.
  - Desteklenen uzantılar: `.pdf`, `.docx`, `.txt`, `.md`

Yerel doküman indeksini oluşturmak/güncellemek için:

```bash
python -m scripts.ingest_local_docs
```

TEKNOFEST sitesini crawl edip indekslemek için:

```bash
python -m scripts.crawl_teknofest
```

İşlemler sonunda Chroma veritabanları:

- `RAG/chroma_local_docs/`
- `RAG/chroma_teknofest_site/`

altına kaydedilecektir.

## Uygulamayı Çalıştırma

```bash
uvicorn app.main:app --reload
```

Ardından tarayıcıdan:

- `http://127.0.0.1:8000/` adresine giderek tek sayfalık chatbot arayüzünü kullanabilirsiniz.

## Testler

```bash
pytest
```

Not: Bazı testler gerçek OpenAI/Tavily anahtarları olmadan veya indeksler oluşturulmamışsa atlanmalı ya da güncellenmelidir; bu repo temel bir başlangıç iskeleti sunar.

