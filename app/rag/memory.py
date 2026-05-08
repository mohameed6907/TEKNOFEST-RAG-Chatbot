"""
memory.py
=========
Conversational memory utilities for the TEKNOFEST RAG pipeline.

Provides:
- REPHRASE_PROMPT: ChatPromptTemplate for standalone question rewriting
- build_rephrase_chain(llm): Returns a runnable rephrase chain
- format_chat_history(messages): Converts SQLite message dicts to a string
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Sen bir soru yeniden yazma asistanısın.
Kullanıcının yeni sorusunu ve geçmiş konuşmayı alarak,
soruyu önceki bağlama ihtiyaç duymadan TAMAMEN ANLAŞILABİLECEK şekilde yeniden yaz.

KRITIK KURALLAR:
1. "Bu kategori", "benim seçtiğim", "önceki seçim", "daha önce belirttiğim" gibi REFERENSLARı 
   geçmişten AÇIKÇA çıkararak yerine yaz
2. TEKNOFEST kategorisi adlarını tamamıyla yaz
3. Özet yapmadan, tüm temel bilgiyi yeniden yapılandır
4. Sadece yeniden yazılmış soruyu döndür, açıklama ekleme

ÖRNEK:
Geçmiş: 
  Kullanıcı: ben sağlıkta yapay zeka alanında yarışacağım
  Asistan: Sağlıkta yapay zeka kategorisinde...

Yeni soru: bu kategoride nelere dikkat etmeliyim
Çıkış: Sağlıkta yapay zeka kategorisinde nelere dikkat etmeliyim""",
    ),
    (
        "human",
        """Konuşma geçmişi:
{chat_history}

Yeni soru: {question}

Bağımsız ve açık soru:""",
    ),
])


def build_rephrase_chain(llm):
    """
    Deterministik rephrase chain (temperature=0 LLM ile kullanılır).
    LLM factory'den purpose='rephrase' ile çağır.

    Returns:
        Runnable: REPHRASE_PROMPT | llm | StrOutputParser
    """
    return REPHRASE_PROMPT | llm | StrOutputParser()


def format_chat_history(messages: list[dict]) -> str:
    """
    SQLite messages tablosundaki [{role, content}] listesini
    okunabilir string'e dönüştür. Son 6 mesajı al (3 tur).

    Args:
        messages: [{"role": "user"|"assistant", "content": "..."}] listesi

    Returns:
        Formatlanmış konuşma geçmişi string'i
    """
    recent = messages[-6:] if len(messages) > 6 else messages
    lines = []
    for msg in recent:
        role = "Kullanıcı" if msg.get("role") == "user" else "Asistan"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)
