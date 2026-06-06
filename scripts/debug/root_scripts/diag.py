import sys
sys.path.insert(0, ".")
from app.config import get_settings
settings = get_settings()

try:
    from langchain_tavily import TavilySearchResults
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults

import inspect
print("TavilySearchResults signature:", inspect.signature(TavilySearchResults.__init__))

tool = TavilySearchResults(max_results=5)
results = tool.invoke("TEKNOFEST insansız kara aracı yarışması ödül")
print(f"\nRaw Tavily results ({len(results)} total):")
for r in results:
    print(f"  url: {r.get('url','')}")
    print(f"  content: {r.get('content','')[:100]}...")
    print()
