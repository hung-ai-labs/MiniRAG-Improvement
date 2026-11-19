"""Historical strict-label Gemini judge with the explicitly pinned model."""
import os
import sys

from .common import JUDGE, setup


if __name__ == '__main__':
    setup()
    os.environ['GEMINI_API_BASE'] = 'https://generativelanguage.googleapis.com/v1beta/openai/'
    for name in ('GEMINI_API_KEY_ONLY', 'MINIRAG_SLM_SEED', 'MINIRAG_MAX_TOKENS', 'GEMINI_RPM'):
        os.environ.pop(name, None)
    import judge_run
    raise SystemExit(judge_run.main(sys.argv[1:] + ['--judgemodel', JUDGE, '--passes', '3',
                                                   '--concurrency', '4', '--invalid-tries', '3']))
