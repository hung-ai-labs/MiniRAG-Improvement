"""Operator-only launcher, matching +CE. Never run by the offline Codex task.

Load the credential path from MINIRAG_DOTENV into the process, then run.main().
"""
import os
import sys

from dotenv import load_dotenv


def main():
    path = os.environ.get('MINIRAG_DOTENV')
    if not path or not os.path.isfile(path):
        raise SystemExit('set MINIRAG_DOTENV to the .env holding the SLM and Gemini keys')
    load_dotenv(path, override=False)
    from reproduce.mhr_set import run
    return run.main()


if __name__ == '__main__':
    raise SystemExit(main())
