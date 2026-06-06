import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

path = 'app/rag/graph.py'
content = open(path, 'r', encoding='utf-8').read()

old = '        bypass_routes = {"llm_knowledge", "kisisel", "direct"}'
new = (
    '        # live_fetch answers are grounded in real-time data: treat as bypass\n'
    '        bypass_routes = {"llm_knowledge", "kisisel", "direct", "live_fetch"}'
)

if old in content:
    content = content.replace(old, new, 1)
    open(path, 'w', encoding='utf-8').write(content)
    print('DONE: bypass_routes updated')
else:
    # Try CRLF variant
    old_crlf = old.replace('\n', '\r\n')
    if old_crlf in content:
        content = content.replace(old_crlf, new.replace('\n', '\r\n'), 1)
        open(path, 'w', encoding='utf-8').write(content)
        print('DONE: bypass_routes updated (CRLF)')
    else:
        print('NOT FOUND - searching context...')
        idx = content.find('bypass_routes')
        print(repr(content[idx-5:idx+80]))
