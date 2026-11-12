#!/bin/zsh
# A-Z: Table 1 (C0 -> + source-time -> + F -> + CE -> + set selection) for one SLM,
# on the 435 held-out questions, 3 generation rounds x 3 DeepSeek judge passes.
# Every step is resumable: rerun the same command after any interruption.
#
#   zsh reproduce/evidence_set/run_ladder.sh qwen
#   zsh reproduce/evidence_set/run_ladder.sh phi      # builds the Phi index first (Phi does the extraction)
set -e
cd "${0:A:h}/../.."
PY=.venv/bin/python
export PYTHONUNBUFFERED=1
SLM=${1:?usage: run_ladder.sh qwen|phi|phi_ropefix}
ROUNDS=${ROUNDS:-3}
WORKERS=${WORKERS:-16}   # concurrent reader requests (the Modal app accepts 32)
export MINIRAG_EVSET_SLM=$SLM
export MINIRAG_LLM_TIMEOUT=240   # a Modal container that dies mid-request must not hang the run

# Modal recycles containers; a step that fails is retried (every step resumes where it stopped).
run_stage () {
  local n=0
  until "$@"; do
    n=$((n+1))
    [ $n -ge 30 ] && { echo "== giving up after $n failures: $*"; return 1; }
    echo "== step failed (attempt $n), retrying in 60 s: $*"
    sleep 60
  done
}

if [ "$SLM" != qwen ]; then
  case $SLM in
    phi)         IDX=LiHua-World-phi-modal;   URLF=phi_url ;;
    phi_ropefix) IDX=LiHua-World-phi-ropefix; URLF=phi_ropefix_url ;;   # LongRoPE-fixed vLLM server
    *) echo "unknown SLM $SLM"; exit 1 ;;
  esac
  if [ ! -f $IDX/INDEX_DONE ]; then
    echo "== index ($SLM on Modal builds its own graph)"
    run_stage env GEMINI_API_BASE="$(cat ~/.config/minirag/$URLF)/v1" GEMINI_API_KEY_ONLY="$(cat ~/.config/minirag/slm_key)" \
        MINIRAG_MAX_TOKENS=1024 GEMINI_RPM=100000 GEMINI_TPM=100000000 \
        $PY reproduce/Step_0_index.py --workingdir ./$IDX --model minirag-slm   # all 442 documents, same as LiHua-World-qwen-modal
    date > $IDX/INDEX_DONE
  fi
  # the scorer is trained on this SLM's own dev200 pools
  run_stage $PY -m reproduce.evidence_set.pipeline capture --split dev200
fi
run_stage $PY -m reproduce.evidence_set.pipeline all --split nondev --rounds $ROUNDS --workers $WORKERS
