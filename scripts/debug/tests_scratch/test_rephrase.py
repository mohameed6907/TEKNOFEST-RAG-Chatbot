"""
Test rephrase node specifically
"""
import asyncio
import sys
sys.path.insert(0, ".")

from app.config import get_settings
from app.rag.memory import build_rephrase_chain, format_chat_history
from app.llm import get_llm_service

async def test_rephrase():
    settings = get_settings()
    llm = get_llm_service(settings).get_chat_model(temperature=0.0, purpose="rephrase")
    rephrase_chain = build_rephrase_chain(llm)
    
    # Simulate chat history from the user's example
    chat_history = [
        {"role": "user", "content": "ben sağlıkta yapay zeka alanında yarışacağım"},
        {"role": "assistant", "content": "Sağlıkta yapay zeka kategorisinde yarışmaya hazırlanıyorsunuz. Bu kategori oldukça kompetitif..."},
        {"role": "user", "content": "teknofestde en sevdiğin kategori ne peki"},
        {"role": "assistant", "content": "En sevdiğim kategori mi? Elbette senin kategorin!"},
    ]
    
    new_questions = [
        "bu kategoride nelere dikkat etmeliyim",
        "robolig yarışması ve benim seçtiğim kategori arasındaki ödül farkı var mı"
    ]
    
    for q in new_questions:
        print(f"\n{'='*60}")
        print(f"Original question: {q}")
        print('='*60)
        
        history_str = format_chat_history(chat_history)
        print(f"History:\n{history_str}\n")
        
        try:
            result = await rephrase_chain.ainvoke({
                "chat_history": history_str,
                "question": q,
            })
            rephrased = result.strip()
            print(f"Rephrased question: {rephrased}")
            
            if "sağlıkta yapay zeka" in rephrased.lower():
                print("✅ CATEGORY CONTEXT PRESERVED")
            else:
                print("❌ CATEGORY CONTEXT LOST")
                
        except Exception as e:
            print(f"ERROR: {e}")
        
        # Add this Q&A to history for next iteration
        chat_history.append({"role": "user", "content": q})
        chat_history.append({"role": "assistant", "content": "Mock response..."})

if __name__ == "__main__":
    asyncio.run(test_rephrase())
