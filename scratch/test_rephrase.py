import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from app.config import get_settings
from app.llm import get_llm_service
from app.rag.memory import build_rephrase_chain

async def test():
    settings = get_settings()
    llm = get_llm_service(settings).get_chat_model(temperature=0.0, purpose="rephrase")
    chain = build_rephrase_chain(llm)
    
    history_str = "Kullanıcı: TEKNOFEST 2026'da kuantum ile alakalı bir yarışma bulunuyor mu?\nAsistan: Evet, TEKNOFEST 2026'da kuantum ile alakalı bir yarışma bulunuyor! Kuantum Teknolojileri Yarışması..."
    question = "peki bu yarışma hakkında daha fazla bilgi verir misin"
    
    print("Running rephrase...")
    res = await chain.ainvoke({
        "chat_history": history_str,
        "question": question
    })
    print("Rephrased:", res)

if __name__ == "__main__":
    asyncio.run(test())
