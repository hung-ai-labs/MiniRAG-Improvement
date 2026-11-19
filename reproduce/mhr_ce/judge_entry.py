"""Deferred strict-label judge launcher: credentials from environment, never .env."""
from __future__ import annotations

import os
import sys

from reproduce.mhr_ce import offline
from reproduce.mhr_st.common import JUDGE


def main():
    offline.setup()
    os.environ['GEMINI_API_BASE'] = 'https://generativelanguage.googleapis.com/v1beta/openai/'
    for name in ('GEMINI_API_KEY_ONLY', 'MINIRAG_SLM_SEED', 'MINIRAG_MAX_TOKENS', 'GEMINI_RPM'):
        os.environ.pop(name, None)
    import judge_run
    return judge_run.main(sys.argv[1:] + ['--judgemodel', JUDGE, '--passes', '3',
                                         '--concurrency', '4', '--invalid-tries', '3'])


if __name__ == '__main__':
    raise SystemExit(main())
