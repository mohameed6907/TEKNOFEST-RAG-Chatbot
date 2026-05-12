import re

file_path = r"C:\Users\myy\.gemini\antigravity\brain\9e996ac7-b70d-446f-ae10-c020b9233c83\.system_generated\steps\215\content.md"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Pattern to find links starting with https://www.teknofest.org/tr/yarismalar/ followed by a slug
pattern = r'\[([^\]]+)\]\(https://www\.teknofest\.org/tr/yarismalar/([a-z0-9\-]+)/\)'
matches = re.findall(pattern, content)

slug_map = {}
for name, slug in matches:
    name = name.strip()
    # Clean up name if it has "Başvuru Tamamlandı" etc. at the end
    name = re.sub(r'\s*Başvuru.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*Henüz.*$', '', name, flags=re.IGNORECASE)
    name = name.strip()
    
    if name and slug and slug not in ["#tabsCategory0", "tabsCategory1", "tabsCategory2", "tabsCategory3", "tabsCategory4", "tabsCategory6", "tabsCategory41", "tabsCategory42"]:
        # Don't overwrite if it already exists, or just keep it
        slug_map[name] = slug

# Now let's generate a dictionary of keywords -> slug
# We can create some common aliases for each
import json

output_map = {}
for name, slug in slug_map.items():
    # Basic keyword mapping: lowercased full name
    lower_name = name.lower()
    
    # We will just print the suggested mappings to manually review and insert into graph.py
    print(f'        "{lower_name}": "{slug}",')

