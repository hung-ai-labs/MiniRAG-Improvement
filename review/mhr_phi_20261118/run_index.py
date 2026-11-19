"""Launcher for the Phi MultiHop-RAG index (PROTOCOL.md, gate I). Runs index_source (MiniRAG d90d892) unchanged.

Keys come from the .env named by MINIRAG_DOTENV (never copied); the Phi endpoint from PHI_SLM_URL, which wins
over the Qwen URL in that .env. MINIRAG_MAX_TOKENS=4096 as for the Qwen index.

    PHI_SLM_URL=https://...-minirag-slm-phi-ropefix-serve.us-east.modal.run \
    MINIRAG_DOTENV=E:/Project/MiniRAG research/MiniRAG/.env \
    python review/mhr_phi_20261118/run_index.py --subset pilot --workingdir indexes/multihop_rag-phi-pilot30
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'index_source'
DOCS = HERE.parents[1] / 'dataset/MultiHop-RAG/processed/docs.jsonl'


def main():
    url = os.environ.get('PHI_SLM_URL', '').strip()
    if 'phi' not in url:
        raise SystemExit('set PHI_SLM_URL to the minirag-slm-phi-ropefix endpoint')
    from dotenv import load_dotenv
    load_dotenv(os.environ['MINIRAG_DOTENV'], override=False)
    os.environ['MINIRAG_SLM_URL'] = url
    os.environ['MINIRAG_MAX_TOKENS'] = '4096'
    os.chdir(SRC)
    sys.path.insert(0, str(SRC / 'reproduce/research'))
    sys.argv = [str(SRC / 'reproduce/research/index_run.py'), '--docs', str(DOCS), *sys.argv[1:]]
    import runpy
    runpy.run_path(str(SRC / 'reproduce/research/index_run.py'), run_name='__main__')


if __name__ == '__main__':
    main()
