import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

sys.stdout.reconfigure(encoding='utf-8')

raw_dir = Path("RAG/raw")
files = list(raw_dir.glob("*"))

for f in files:
    if f.suffix.lower() == ".pdf":
        try:
            loader = PyPDFLoader(str(f))
            docs = loader.load()
            text = " ".join([d.page_content for d in docs])
        except Exception as e:
            print(f"Error reading {f.name}: {e}")
            continue
    elif f.suffix.lower() == ".docx":
        try:
            loader = Docx2txtLoader(str(f))
            docs = loader.load()
            text = " ".join([d.page_content for d in docs])
        except Exception as e:
            print(f"Error reading {f.name}: {e}")
            continue
    else:
        continue

    count = text.lower().count("blokzincir")
    if count > 0:
        print(f"Found 'blokzincir' {count} times in {f.name}")
