"""Operator launcher for the online +CE steps (answers / execute / finish).

reproduce.mhr_ce.run never loads credential files. This launcher loads the SLM and Gemini keys the
same way the +ST/+F runs did (reproduce.mhr_st.common.setup): from the .env named by MINIRAG_DOTENV
(another checkout's file, never copied), into this process only; the answer and judge subprocesses
inherit them through the environment.

    MINIRAG_DOTENV=E:/Project/MiniRAG research/MiniRAG/.env python -m reproduce.mhr_ce.online execute
    MINIRAG_DOTENV=... python -m reproduce.mhr_ce.online finish --rounds 20
"""
import os
import sys

from dotenv import load_dotenv


def main():
    path = os.environ.get('MINIRAG_DOTENV')
    if not path or not os.path.isfile(path):
        raise SystemExit('set MINIRAG_DOTENV to the .env holding the SLM and Gemini keys')
    load_dotenv(path, override=False)
    from reproduce.mhr_ce import run
    return run.main()


if __name__ == '__main__':
    raise SystemExit(main())
