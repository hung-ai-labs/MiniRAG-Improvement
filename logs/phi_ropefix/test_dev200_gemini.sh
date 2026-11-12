#!/bin/zsh
# Test before the full run: Table 1 ladder on dev200, 1 generation round x 3 Gemini judge passes.
cd /Users/hunggoodboy/Documents/HTTM-TK/MiniRAG-HTTM
export PYTHONUNBUFFERED=1 MINIRAG_EVSET_SLM=phi_ropefix MINIRAG_LLM_TIMEOUT=240 MINIRAG_EVSET_JUDGE=gemini
PY=.venv/bin/python
while pgrep -f "pipeline generate --split dev200" >/dev/null; do sleep 20; done   # the generation already running
run_stage () { local n=0; until "$@"; do n=$((n+1)); [ $n -ge 20 ] && { echo "== giving up: $*"; return 1; }; echo "== step failed ($n), retry in 60 s"; sleep 60; done; }
for s in generate judge report; do
  run_stage $PY -m reproduce.evidence_set.pipeline $s --split dev200 --rounds 1 --workers 16 --judge-workers 8 || exit 1
done
echo "== TEST DONE"
