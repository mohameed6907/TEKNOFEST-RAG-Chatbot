import json
lines = open('RAG/eval_log.jsonl','r',encoding='utf-8').readlines()
last5 = [json.loads(l) for l in lines[-5:]]
for e in last5:
    q = e.get('query','')[:60]
    print(f"route={e.get('route')} | hal={e.get('hallucination_status')} | ret={e.get('retrieved_count')} | sel={e.get('selected_count')} | q={q}")
