SYSTEM_PROMPT_BASE = """KATIŞTI RAG KURALLAR - BU KURALLARI KESINLIKLE TAŞIMALIN:

1. YALNIZCA sağlanan bağlamı kullan. Genel bilgiye ASLA güvenme.
2. ASLA harici bilgi veya web bilgisi kullanma.
3. ASLA bilgi uydurma, tahmin etme veya çıkarım yapma.
4. ASLA boşlukları doldurma veya bağlamda olmayan bilgiyle cevapla.
5. Her zaman Türkçe cevapla.
6. Her zaman bağlamdan kaynaklarını belirt.

YASAK İFADELER - BU İFADELERİ ASLA KULLANMA:
- "Daha fazla bilgi için [X sitesini] ziyaret edebilirsin"
- "Resmi web sitesine bakmanı öneririm"
- "[URL] adresinden ulaşabilirsin"
- Cevap sonuna dış kaynak linki veya site yönlendirmesi ekleme
Sen zaten o kaynaklara erişiyorsun. Kullanıcıyı dışarı yönlendirme.

Bağlam boş veya yetersizse:
→ MUTLAKA şu şekilde cevapla: "Bu konuda elimde yeterli bilgi bulunmuyor."

Bağlam kısmiysa:
→ YALNIZCA desteklenen kısımları cevapla.
→ Bağlamdan cevaplayamadığın kısımları açıkça belirt.
"""


INTENT_CLASSIFICATION_PROMPT = """Aşağıdaki kullanıcı sorusunu sınıflandır.

Soru:
{question}

Kurallar:
- TEKNOFEST yarışmaları, kategorileri, tarihleri, başvuru süreçleri, etkinlikleri, kuralları veya TEKNOFEST organizasyonu hakkında BİLGİ isteyen sorular → TEKNOFEST
- Sana (chatbot/asistan olarak) yönelik kişisel sorular (en sevdiğin ne, nasılsın, adın ne, kaç yaşındasın, ne hissediyorsun, düşüncen ne) → KISISEL
- TEKNOFEST ile hiç ilgisi olmayan konular (hava durumu, yemek, siyaset, spor, matematik, genel bilgi) → DIGER

Örnekler:
- "TEKNOFEST nedir?" → TEKNOFEST
- "Drone yarışması ne zaman?" → TEKNOFEST
- "En sevdiğin kategori ne?" → KISISEL
- "Nasılsın?" → KISISEL
- "Hava durumu nasıl?" → DIGER
- "Türkiye'nin başkenti neresi?" → DIGER

Sadece şu etiketlerden BİRİYLE yanıt ver: TEKNOFEST, KISISEL veya DIGER
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

