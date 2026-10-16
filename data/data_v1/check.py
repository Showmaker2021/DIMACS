import json, hashlib
rows = [json.loads(l) for l in open('manifest.jsonl')]
key = sorted((r['family_id'], r['split'], r['seed'], r['sha256']) for r in rows)
print(len(rows), hashlib.sha256(json.dumps(key).encode()).hexdigest())