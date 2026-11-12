#!/bin/zsh
# Test before the full run: Table 1 ladder on dev200 (cross-fitted scores), 1 generation round x 3 DeepSeek passes.
cd /Users/hunggoodboy/Documents/HTTM-TK/MiniRAG-HTTM
export PYTHONUNBUFFERED=1 MINIRAG_EVSET_SLM=phi_ropefix MINIRAG_LLM_TIMEOUT=240
PY=.venv/bin/python
while pgrep -f "pipeline capture --split dev200" >/dev/null; do sleep 20; done   # let the dev200 capture finish
run_stage () { local n=0; until "$@"; do n=$((n+1)); [ $n -ge 20 ] && { echo "== giving up: $*"; return 1; }; echo "== step failed ($n), retry in 60 s"; sleep 60; done; }
run_stage $PY -m reproduce.evidence_set.pipeline capture --split dev200
for s in score build generate judge report; do
  run_stage $PY -m reproduce.evidence_set.pipeline $s --split dev200 --rounds 1 --workers 16 || exit 1
done
echo "== TEST DONE"
