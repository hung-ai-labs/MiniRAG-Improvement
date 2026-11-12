"""Kiểm sức khoẻ một index MiniRAG ngay sau khi dựng (dataset bất kỳ, SLM bất kỳ). 0 API.

Đếm các chunk không trích được thực thể nào, chia theo độ dài chunk. Đây là triệu chứng của
lỗi LongRoPE khi phục vụ Phi-3.5 trên vLLM (xem modal_slm_phi.py). Lỗi đánh vào chunk dài
trước, vì prompt trích xuất của chúng vượt mốc 4.096 vị trí.

    python -m reproduce.index_health ./LiHua-World-phi-ropefix
    python -m reproduce.index_health ./my-new-index --max-zero 0.05   # exit 1 nếu > 5% chunk rỗng

Số đo trên LiHua-World:
    Phi, server gốc (lỗi)     167/516 chunk rỗng (32%); chunk >= 800 token: 85/116 rỗng  (lúc chẩn đoán, index đã huỷ)
    Phi, server đã vá           4/516 rỗng (0,8%); chunk >= 800 token: 3/116
    Qwen2.5-3B                  9/514 rỗng (1,8%)
"""
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKETS = ((0, 300), (300, 800), (800, 10 ** 9))


def health(workdir):
    workdir = Path(workdir)
    chunks = json.loads((workdir / 'kv_store_text_chunks.json').read_text())
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    root = ET.parse(workdir / 'graph_chunk_entity_relation.graphml').getroot()
    keys = {k.get('id'): (k.get('for'), k.get('attr.name')) for k in root.findall('g:key', ns)}
    covered, nodes, edges = set(), 0, 0
    for el in root.iter():
        tag = el.tag.split('}')[-1]
        nodes += tag == 'node'
        edges += tag == 'edge'
        if tag == 'data' and keys.get(el.get('key'), (None, None))[1] == 'source_id' and el.text:
            covered.update(s for s in re.split(r'<SEP>', el.text) if s.startswith('chunk-'))
    rows = []
    for lo, hi in BUCKETS:
        ids = [c for c, r in chunks.items() if lo <= r.get('tokens', 0) < hi]
        rows.append((lo, hi, len(ids), sum(c not in covered for c in ids)))
    zero = sum(c not in covered for c in chunks)
    return {'chunks': len(chunks), 'zero_entity_chunks': zero, 'nodes': nodes, 'edges': edges, 'by_length': rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('workdir')
    ap.add_argument('--max-zero', type=float, default=0.05, help='tỷ lệ chunk rỗng tối đa chấp nhận (mặc định 5%%)')
    a = ap.parse_args()
    h = health(a.workdir)
    rate = h['zero_entity_chunks'] / max(h['chunks'], 1)
    print(f"{a.workdir}: {h['chunks']} chunk, {h['nodes']} node, {h['edges']} cạnh, "
          f"{h['zero_entity_chunks']} chunk không có thực thể ({100 * rate:.1f}%)")
    for lo, hi, n, z in h['by_length']:
        label = f'>= {lo}' if hi > 10 ** 8 else f'{lo}-{hi}'
        print(f'  chunk {label:>8} token: {z}/{n} rỗng')
    long_n, long_z = h['by_length'][-1][2:]
    if rate > a.max_zero or (long_n and long_z / long_n > 2 * a.max_zero):
        print('CẢNH BÁO: index yếu. Nếu là Phi-3 / Phi-3.5 trên vLLM, kiểm tra server đã bật ROPE_FIX '
              '(reproduce/modal_slm_phi.py, hoặc reproduce/phi_longrope_fix.py khi tự chạy vLLM).')
        sys.exit(1)
    print('OK')


if __name__ == '__main__':
    main()
