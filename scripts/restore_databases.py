import os
import subprocess
import sys

def restore():
    print("Database'ler stash@{0}'dan geri yükleniyor...")
    
    paths = {
        "RAG/chroma_local_docs/chroma.sqlite3": "stash@{0}:RAG/chroma_local_docs/chroma.sqlite3",
        "RAG/chroma_teknofest_site/chroma.sqlite3": "stash@{0}:RAG/chroma_teknofest_site/chroma.sqlite3"
    }
    
    success = True
    for dest, src in paths.items():
        try:
            # Get the content from git stash
            content = subprocess.check_output(["git", "show", src])
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            
            # Write the file
            with open(dest, "wb") as f:
                f.write(content)
            print(f"Başarıyla geri yüklendi: {dest} ({len(content)} byte)")
        except subprocess.CalledProcessError as e:
            print(f"Hata: Git stash'tan veri alınamadı. {e}")
            success = False
        except PermissionError:
            print(f"\nHATA: '{dest}' dosyası kilitli!")
            print("Lütfen çalışan uvicorn ve langgraph sunucularını durdurun ve bu betiği tekrar çalıştırın.\n")
            success = False
            break
        except Exception as e:
            print(f"Hata oluştu: {e}")
            success = False
            break

    if success:
        print("\nVeritabanları başarıyla geri yüklendi! Sunucularınızı yeniden başlatabilirsiniz.")
    else:
        sys.exit(1)

if __name__ == "__main__":
    restore()
