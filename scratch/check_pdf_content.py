import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

sys.stdout.reconfigure(encoding='utf-8')

pdf_path = Path("RAG/raw/TEKNOFEST_2026_Ansiklopedi.pdf")
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

for i, doc in enumerate(docs):
    text = doc.page_content
    count = text.lower().count("blokzincir")
    if count > 0:
        print(f"Page {i+1} has 'blokzincir' {count} times. Content snippet:")
        print(text[:300].replace('\n', ' '))
        print("-" * 50)
