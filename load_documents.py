#!/usr/bin/env python
"""
Chroma collection'larına dokümanları yükleme scripti.
RAG/raw dizininden PDF, DOCX, TXT dosyalarını okuyup Chroma'ya yükler.
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path
import logging
from app.config import get_settings
from app.rag.embedding_service import get_embedding_service
import chromadb
from chromadb.config import Settings as ChromaSettings

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Document loaders
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        UnstructuredWordDocumentLoader,
    )
except ImportError:
    try:
        from langchain.document_loaders import (
            PyPDFLoader,
            TextLoader,
            UnstructuredWordDocumentLoader,
        )
    except ImportError:
        print("❌ HATA: langchain document loaders bulunamadı")
        print("pip install langchain-community pypdf python-docx unstructured yapmalısınız")
        sys.exit(1)

from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(raw_dir: Path):
    """RAG/raw dizininden dokümanları yükle"""
    documents = []

    print(f"\n📂 Dokümanlar yükleniyor: {raw_dir}")
    print("=" * 70)

    # Dosya türlerine göre yükle
    for file_path in sorted(raw_dir.iterdir()):
        if not file_path.is_file():
            continue

        try:
            if file_path.suffix.lower() == '.pdf':
                print(f"📄 PDF yükleniyor: {file_path.name}")
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
                print(f"   ✓ {len(docs)} sayfa yüklendi")

            elif file_path.suffix.lower() in ['.docx', '.doc']:
                print(f"📝 DOCX yükleniyor: {file_path.name}")
                loader = UnstructuredWordDocumentLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
                print(f"   ✓ {len(docs)} blok yüklendi")

            elif file_path.suffix.lower() == '.txt':
                print(f"📋 TXT yükleniyor: {file_path.name}")
                loader = TextLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
                print(f"   ✓ {len(docs)} satır yüklendi")

        except Exception as e:
            print(f"   ⚠️  HATA: {e}")
            continue

    print(f"\n✓ Toplam {len(documents)} document blok yüklendi")
    return documents

def chunk_documents(documents, settings):
    """Dokümanları chunk'la"""
    print(f"\n📏 Chunking yapılıyor...")
    print("=" * 70)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_target_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"✓ {len(chunks)} chunk oluşturuldu")
    print(f"  • Chunk boyutu: {settings.chunk_target_size}")
    print(f"  • Overlap: {settings.chunk_overlap}")

    return chunks

def upload_to_chroma(chunks, settings):
    """Chunk'ları Chroma'ya yükle"""
    print(f"\n🚀 Chroma'ya yükleniyor...")
    print("=" * 70)

    # Embedding service
    embed_svc = get_embedding_service(settings)
    print(f"✓ Embedding provider: {settings.embedding_provider}")
    print(f"✓ Embedding model: {settings.embedding_model_name}")

    # Chroma client
    client = chromadb.PersistentClient(
        path=str(settings.rag_root / "chroma_local_docs"),
        settings=ChromaSettings(allow_reset=True)
    )

    # Collection'a yükle
    collection = client.get_collection(settings.chroma_local_collection)

    print(f"\n📤 {len(chunks)} chunk Chroma'ya ekleniyor...")

    # Batch upload (performans için)
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        ids = [f"doc_{j}" for j in range(i, i+len(batch))]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata or {} for doc in batch]

        try:
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )
            print(f"  ✓ {i+len(batch)}/{len(chunks)} chunk yüklendi")
        except Exception as e:
            print(f"  ❌ HATA batch {i//batch_size+1}: {e}")
            return False

    # Doğrulama
    final_count = collection.count()
    print(f"\n✅ Upload tamamlandı!")
    print(f"✓ Collection'da toplam {final_count} chunk var")

    return True

def main():
    print("=" * 70)
    print("CHROMA COLLECTION'LARINA DOKÜMAN YÜKLEME")
    print("=" * 70)

    settings = get_settings()
    raw_dir = settings.rag_root / "raw"

    if not raw_dir.exists():
        print(f"❌ HATA: {raw_dir} dizini bulunamadı!")
        return False

    # Adım 1: Dokümanları yükle
    documents = load_documents(raw_dir)
    if not documents:
        print("❌ HATA: Hiç dokument yüklenemedi!")
        return False

    # Adım 2: Chunk'la
    chunks = chunk_documents(documents, settings)
    if not chunks:
        print("❌ HATA: Chunk oluşturulamadı!")
        return False

    # Adım 3: Chroma'ya yükle
    success = upload_to_chroma(chunks, settings)

    if success:
        print("\n" + "=" * 70)
        print("✅ BAŞARILI: Dokümanlar Chroma'ya yüklendi!")
        print("=" * 70)
        print("\n🧪 Şimdi test sorgusu deneyin:")
        print("   Soru: 'TEKNOFEST Robolig yarışması hakkında ne biliyorsun?'")
        return True
    else:
        print("\n❌ Upload başarısız oldu")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
