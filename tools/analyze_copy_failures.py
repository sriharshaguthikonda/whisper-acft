import json
from collections import Counter
from pathlib import Path
p=Path(r"J:\kaggle_publish\acft-moonshine-Record_chunks_publish\copy_failures.jsonl")
c1=Counter()
bad=0
samples=[]
for line in p.open('r',encoding='utf-8',errors='replace'):
    line=line.strip()
    if not line:
        continue
    try:
        o=json.loads(line)
    except Exception:
        bad += 1
        if len(samples)<5:
            samples.append(line)
        continue
    err=str(o.get('error',''))
    c1[err.split(':',1)[0]] += 1
print('parsed', sum(c1.values()), 'bad_lines', bad)
print('types', c1.most_common(10))
if samples:
    print('sample_bad_lines:')
    for s in samples:
        print(s)
