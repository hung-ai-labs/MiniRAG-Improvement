"""Refuse an online run that would reach or bill a retired Modal workspace.

Modal picks the workspace from the active profile in ~/.modal.toml (or MODAL_PROFILE),
and every SLM endpoint URL names its workspace as ``https://<workspace>--<app>-serve...``.
After a workspace runs out of credit both can silently keep pointing at the old account.
This check pins them to one workspace before any request is sent.

Set these in the credential file named by MINIRAG_DOTENV (kept outside the repository)
or in the environment:

    MODAL_WORKSPACE=<workspace that should pay for this run>
    MODAL_RETIRED_WORKSPACES=<old workspace>,<another old workspace>

Check the current setup without sending a request:

    python -B -m reproduce.modal_account [URL ...]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from urllib.parse import urlparse

_KEYS = ("MODAL_WORKSPACE", "MODAL_RETIRED_WORKSPACES")


def _load_settings():
    missing = [key for key in _KEYS if not os.environ.get(key)]
    path = os.environ.get("MINIRAG_DOTENV")
    if missing and path and os.path.exists(path):
        from dotenv import dotenv_values

        values = dotenv_values(path)
        for key in missing:
            if values.get(key):
                os.environ[key] = values[key]
    workspace = os.environ.get("MODAL_WORKSPACE", "").strip().lower()
    if not workspace:
        raise SystemExit("Set MODAL_WORKSPACE to the Modal workspace that should serve and pay for this run")
    retired = {name.strip().lower() for name in os.environ.get("MODAL_RETIRED_WORKSPACES", "").split(",")
               if name.strip()}
    if workspace in retired:
        raise SystemExit(f"MODAL_WORKSPACE={workspace} is listed in MODAL_RETIRED_WORKSPACES")
    return workspace, retired


def url_workspace(url):
    host = (urlparse(url.strip()).hostname or "").lower()
    if not host.endswith(".modal.run") or "--" not in host:
        return None
    return host.split("--", 1)[0]


def active_profile():
    """Return (profile, workspace) that the Modal client will use."""
    if os.environ.get("MODAL_TOKEN_ID") or os.environ.get("MODAL_TOKEN_SECRET"):
        raise SystemExit("Unset MODAL_TOKEN_ID/MODAL_TOKEN_SECRET and select the workspace with a Modal profile")
    result = subprocess.run([sys.executable, "-m", "modal", "profile", "list", "--json"],
                            capture_output=True, text=True, encoding="utf-8", timeout=180)
    if result.returncode != 0:
        raise SystemExit("Could not list Modal profiles: " + (result.stderr or result.stdout).strip()[-500:])
    profiles = json.loads(result.stdout)
    wanted = os.environ.get("MODAL_PROFILE")
    for entry in profiles:
        if entry["name"] == wanted if wanted else entry.get("active"):
            return entry["name"], (entry.get("workspace") or "").lower()
    raise SystemExit(f"Modal profile {wanted!r} is not configured" if wanted else "No active Modal profile")


def check(urls):
    """Verify the endpoints in ``urls`` ({label: url}) and the Modal client use MODAL_WORKSPACE.

    Pins MODAL_PROFILE in the environment so child processes keep the verified profile."""
    workspace, retired = _load_settings()
    for label, url in urls.items():
        owner = url_workspace(url)
        if owner is None:
            raise SystemExit(f"{label} is not a Modal endpoint URL: {url}")
        if owner in retired:
            raise SystemExit(f"{label} points at retired Modal workspace {owner}: {url}")
        if owner != workspace:
            raise SystemExit(f"{label} points at Modal workspace {owner}, expected {workspace}: {url}")
    profile, active = active_profile()
    if active in retired:
        raise SystemExit(f"Modal profile {profile} belongs to retired workspace {active}; "
                         f"run `modal profile activate <profile of {workspace}>` or set MODAL_PROFILE")
    if active != workspace:
        raise SystemExit(f"Modal profile {profile} uses workspace {active}, expected {workspace}")
    os.environ["MODAL_PROFILE"] = profile
    print(f"Modal workspace {workspace} (profile {profile}); endpoints: {', '.join(urls) or 'none'}", flush=True)
    return workspace


if __name__ == "__main__":
    check({f"url{index}": url for index, url in enumerate(sys.argv[1:], 1)}
          | {key: os.environ[key] for key in ("MINIRAG_SLM_URL", "PHI_SLM_URL") if os.environ.get(key)})
