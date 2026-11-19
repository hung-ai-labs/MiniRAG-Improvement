"""Run a ladder module as __main__ with the Phi profile and inherited credentials."""

import os
from pathlib import Path
import runpy
import subprocess
import sys

from . import profile

MODULE = 'reproduce.mhr_phi.launch'
C0_MODULE = 'reproduce.mhr_phi.c0'
SECONDARY_MODULE = 'reproduce.mhr_phi.secondary'
_INHERITED = '_MHR_PHI_LAUNCH_ENV'
_OFFLINE_ONLY = '_MHR_PHI_OFFLINE_ONLY'
_ONLINE = {
    C0_MODULE: {'keywords', 'answers', 'execute'},
    SECONDARY_MODULE: {'ce-drift'},
    'reproduce.mhr_st.run': {'preflight', 'answers', 'execute'},
    'reproduce.mhr_f.run': {'answers', 'execute', 'finish'},
    'reproduce.mhr_ce.run': {'answers', 'execute', 'finish'},
    'reproduce.mhr_set.run': {'answers', 'execute', 'finish'},
}
_ALWAYS_ONLINE = {
    'reproduce.mhr_st.judge', 'reproduce.mhr_st.finish', 'reproduce.mhr_ce.scores',
    'reproduce.mhr_ce.judge_entry', 'reproduce.mhr_set.judge_entry',
    'reproduce.mhr_ce.online', 'reproduce.mhr_set.online',
}


def _profiled(module):
    return isinstance(module, str) and (profile.is_ladder(module) or module in (C0_MODULE, SECONDARY_MODULE))


def rewrite_command(command):
    if (not isinstance(command, (list, tuple)) or not command
            or not isinstance(command[0], (str, os.PathLike))
            or Path(command[0]).resolve() != Path(sys.executable).resolve()):
        return command
    # Only inspect interpreter options. A script may have its own unrelated -m.
    arguments = list(command)
    index = 1
    while index < len(arguments):
        option = arguments[index]
        if not isinstance(option, str) or not option.startswith('-') or option in ('-', '--', '-c'):
            break
        if option == '-m':
            if index + 1 < len(arguments) and _profiled(arguments[index + 1]):
                return arguments[:index + 1] + [MODULE] + arguments[index + 1:]
            break
        index += 2 if option in ('-W', '-X') else 1
    return command


def install_subprocess_rewrite():
    """subprocess.run and check_output also go through subprocess.Popen."""
    original = subprocess.Popen
    if getattr(original, '_phi_launcher', False):
        return

    class ProfiledPopen(original):
        _phi_launcher = True

        def __init__(self, args, *positional, **kwargs):
            rewritten = rewrite_command(args)
            if rewritten is not args:
                env = kwargs.get('env')
                kwargs['env'] = dict(os.environ if env is None else env) | {_INHERITED: '1'}
                kwargs['env']['PYTHONDONTWRITEBYTECODE'] = '1'
                if profile._network_denied:
                    kwargs['env'][_OFFLINE_ONLY] = '1'
            super().__init__(rewritten, *positional, **kwargs)

    subprocess.Popen = ProfiledPopen


def _is_online(module, args):
    if any(arg in ('-h', '--help') for arg in args):
        return False
    return module in _ALWAYS_ONLINE or any(arg in _ONLINE.get(module, ()) for arg in args)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('usage: python -B -m reproduce.mhr_phi.launch <module> [args]')
        return 0
    module, *args = sys.argv[1:]
    if not _profiled(module):
        raise SystemExit(f'Unsupported Phi ladder module: {module}')
    if Path.cwd().resolve() != profile.ROOT:
        raise SystemExit(f'Run the Phi launcher from the repository root: {profile.ROOT}')

    online = _is_online(module, args)
    inherited = os.environ.pop(_INHERITED, None) == '1'
    if online and os.environ.pop(_OFFLINE_ONLY, None) == '1':
        raise SystemExit('An offline Phi command cannot launch an online stage')
    import dotenv
    load_dotenv = dotenv.load_dotenv
    profile.apply(offline=not online)
    if online and not inherited:
        dotenv_path = os.environ.get('MINIRAG_DOTENV')
        if not dotenv_path:
            raise SystemExit('Set MINIRAG_DOTENV to the credential file before running online commands')
        with profile._launcher_dotenv(dotenv_path) as path:
            load_dotenv(path, override=False)
    url = os.environ.get('PHI_SLM_URL', '').strip()
    if online and not url:
        raise SystemExit('Set PHI_SLM_URL to the Phi LongRoPE serving endpoint')
    if url:
        os.environ['MINIRAG_SLM_URL'] = url
    if 'PYTHONUTF8' not in os.environ:
        os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    install_subprocess_rewrite()
    sys.argv = [module, *args]
    runpy.run_module(module, run_name='__main__', alter_sys=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
