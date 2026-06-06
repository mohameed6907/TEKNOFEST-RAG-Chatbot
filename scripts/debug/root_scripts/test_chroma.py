import sys, chromadb
sys.path.insert(0, ".")
from app.config import get_settings
from app.rag.embedding_service import EmbeddingService
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

settings = get_settings()
svc = EmbeddingService(settings)
emb = svc.embed_query("insansız kara aracı yarışması ödül birinci")

client = chromadb.PersistentClient(path="RAG/chroma_teknofest_site")
cols = client.list_collections()
if cols:
    r = cols[0].query(query_embeddings=[emb], n_results=3, include=["documents","metadatas","distances"])
    for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
        print(f"dist={dist:.3f} | source={meta.get('source','?')}")
        print(f"content: {doc[:300]}")
        print()
else:
    print("No collections found in chroma_teknofest_site")
