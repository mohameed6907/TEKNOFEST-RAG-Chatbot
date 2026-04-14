SYSTEM_PROMPT_BASE = """Sen TEKNOFEST odaklı bir yardımcı asistansın.
Kullanıcıya mümkün olduğunca Türkçe yanıt ver.
Eğer TEKNOFEST ile ilgili olmayan genel bir soruysa, genel bilgi sağlayabilirsin.
Eğer TEKNOFEST ile ilgili bir soruysa:
- Önce yerel dokümanlar ve TEKNOFEST web içeriği gibi kaynaklardan gelen metinleri kullan.
- Cevabında mutlaka hangi kaynaklardan yararlandığını özetle.
- Kaynaklarda olmayan bilgileri uydurma. Emin değilsen, emin olmadığını açıkça söyle.
Güvenilir bilgi yoksa şu ifadeyi kullan: "Insufficient reliable information available."
"""


INTENT_CLASSIFICATION_PROMPT = """Aşağıdaki kullanıcı sorusunun TEKNOFEST ile ilgili olup olmadığını sınıflandır.

Soru:
{question}

Sadece şu etiketlerden biriyle yanıt ver:
- TEKNOFEST
- DIGER
"""


HALLUCINATION_CHECK_PROMPT = """Aşağıdaki cevap ve verilen kaynak özetlerine göre,
cevabın kaynaklarla çelişip çelişmediğini veya bariz şekilde uydurma içerik içerip içermediğini değerlendir.

Soru:
{question}

Cevap:
{answer}

Kaynak özetleri:
{source_summaries}

Sadece şu etiketlerden biriyle yanıt ver:
- GUVENLI
- SUPHELI
"""

