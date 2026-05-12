import asyncio
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import get_settings
from app.llm import get_llm_service

REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a question rewriting assistant for a TEKNOFEST chatbot.
Given the conversation history (which includes both user questions and assistant responses) 
and a new short or ambiguous question, rewrite it into a fully self-contained question.

Pay special attention to:
- Short questions like "more about this", "what about that", "is there more" — 
  these refer to the topic discussed in the immediately preceding assistant response
- Single words or pronouns that reference something the assistant mentioned
- Follow-up questions that only make sense in context of the previous answer

CRITICAL: If the assistant was talking about a specific competition (e.g., "İnsansız Kara Aracı (İKA) Yarışması"), 
you MUST include the full name of that competition in the rewritten question to make it standalone.

Return only the rewritten question in Turkish. Nothing else."""),
    ("human", """Conversation history:
{chat_history}

New question: {question}

Rewritten standalone question:""")
])

async def test():
    settings = get_settings()
    llm = get_llm_service(settings).get_chat_model(temperature=0.0)
    chain = REPHRASE_PROMPT | llm | StrOutputParser()
    
    history = """Kullanıcı: İnsansız Kara Aracı (İKA) yarışması hakkında bilgi verir misin?
Asistan (özet): İnsansız Kara Aracı (İKA) Yarışması, TEKNOFEST kapsamında düzenlenen bir yarışmadır. Bu yarışmada araçların parkur üzerindeki performansı ölçülür. Parkur çeşitli engellerden oluşmaktadır."""
    
    question = "parkur hakkında daha fazla bilgi var mı"
    
    res = await chain.ainvoke({"chat_history": history, "question": question})
    print(f"History:\n{history}")
    print(f"Question: {question}")
    print(f"Result: {res}")

if __name__ == "__main__":
    asyncio.run(test())
