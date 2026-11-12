"""A/B: Phi-3.5 extraction on the original vLLM server vs the LongRoPE-fixed one (12/11/2026).

Same calls MiniRAG makes while indexing (operate.py extract_entities): the entity_extraction
prompt, one gleaning round with history, max_tokens 1024, server-default sampling, 32 requests
in flight (the index's concurrency), long and short chunks mixed as in the real run.
Output: review/phi_rope_ab_20261112/{arm}.jsonl (raw text kept) and summary.json.
"""
import json
import random
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import importlib.util
_s = importlib.util.spec_from_file_location('prompt', ROOT / 'minirag/prompt.py')
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
PROMPTS = _m.PROMPTS
OUT = ROOT / 'review/phi_rope_ab_20261112'
# Modal URLs of the two deployments (modal_slm_phi.py with ROPE_FIX = False / True)
SERVERS = {'original': (Path.home() / '.config/minirag/phi_url').read_text().strip().rstrip('/'),
           'ropefix': (Path.home() / '.config/minirag/phi_ropefix_url').read_text().strip().rstrip('/')}
KEY = (Path.home() / '.config/minirag/slm_key').read_text().strip()
TD, RD, CD = PROMPTS['DEFAULT_TUPLE_DELIMITER'], PROMPTS['DEFAULT_RECORD_DELIMITER'], PROMPTS['DEFAULT_COMPLETION_DELIMITER']
BASE = dict(tuple_delimiter=TD, record_delimiter=RD, completion_delimiter=CD,
            entity_types=','.join(PROMPTS['DEFAULT_ENTITY_TYPES']))
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def chat(url, messages):
    body = json.dumps({'model': 'minirag-slm', 'messages': messages, 'max_tokens': 1024}).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request(url + '/v1/chat/completions', body,
                                         {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + KEY})
            with opener.open(req, timeout=900) as r:
                j = json.loads(r.read())
            return j['choices'][0]['message']['content'] or '', j['usage']
        except Exception as e:  # noqa: BLE001  cold start / transient
            print('retry', attempt, type(e).__name__, e, flush=True)
    raise RuntimeError('server failed')


def parse(text):
    ents = rels = 0
    for rec in re.split('|'.join(map(re.escape, [RD, CD])), text):
        m = re.search(r'\((.*)\)', rec)
        if not m:
            continue
        a = m.group(1).split(TD)
        ents += len(a) >= 4 and a[0] == '"entity"'
        rels += len(a) >= 5 and a[0] == '"relationship"'
    return ents, rels


def one(url, cid, content):
    prompt = PROMPTS['entity_extraction'].format(**BASE, input_text=content)
    out, u1 = chat(url, [{'role': 'user', 'content': prompt}])
    glean, u2 = chat(url, [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': out},
                           {'role': 'user', 'content': PROMPTS['entiti_continue_extraction']}])
    e, r = parse(out + glean)
    return dict(chunk=cid, prompt_tokens=u1['prompt_tokens'], out_tokens=u1['completion_tokens'],
                glean_tokens=u2['completion_tokens'], entities=e, relations=r, raw=out, glean=glean)


def main(n_long=48, n_short=48):
    OUT.mkdir(parents=True, exist_ok=True)
    P = json.loads((ROOT / 'LiHua-World-phi-modal/kv_store_text_chunks.json').read_text())
    rng = random.Random(20261112)
    long_ = rng.sample(sorted(c for c in P if P[c]['tokens'] >= 800), n_long)
    short = rng.sample(sorted(c for c in P if P[c]['tokens'] < 300), n_short)
    sel = long_ + short
    rng.shuffle(sel)
    summary = {}
    for arm in sys.argv[1:] or list(SERVERS):
        with ThreadPoolExecutor(32) as pool:
            rows = list(pool.map(lambda c: one(SERVERS[arm], c, P[c]['content']), sel))
        with (OUT / f'{arm}.jsonl').open('w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        s = {}
        for name, ids in (('long>=800', set(long_)), ('short<300', set(short))):
            rs = [r for r in rows if r['chunk'] in ids]
            s[name] = dict(n=len(rs), zero_entity=sum(r['entities'] == 0 for r in rs),
                           entities=sum(r['entities'] for r in rs), relations=sum(r['relations'] for r in rs))
        summary[arm] = s
        print(arm, json.dumps(s), flush=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
