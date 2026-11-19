# Phi MultiHop-RAG profile and C0 verification

The Phi profile, C0 runner, and operator runbook are ready for the offline-verified scope. The task ran with the required MiniRAG interpreter from the repository root and `PYTHONUTF8=1`. All executed commands were local. Credential files, SLM generation, Gemini judging, and Modal scoring were outside the executed scope.

## Added files

| File | Purpose |
|---|---|
| [profile.py](../../reproduce/mhr_phi/profile.py) | Selects Phi artifact paths and report identities before frozen ladder code executes. Installs the Qwen write guard and the credential-file audit. |
| [launch.py](../../reproduce/mhr_phi/launch.py) | Runs ladder modules as `__main__` and passes the profile to answer, context, and judge children. |
| [runtime.py](../../reproduce/mhr_phi/runtime.py) and [c0.py](../../reproduce/mhr_phi/c0.py) | Use the frozen C0 phases, three-worker retrieval, per-request tracing, and same-seed receipts and checkpoints. |
| [secondary.py](../../reproduce/mhr_phi/secondary.py) | Points the registered CE secondary checks at the Phi outputs in memory. |
| [test_profile.py](../../reproduce/mhr_phi/test_profile.py) and [RUNBOOK.md](../../reproduce/mhr_phi/RUNBOOK.md) | Offline profile tests and the exact operator command sequence for C0 through SET. |
| [serving_phi_ropefix_modal.json](serving_phi_ropefix_modal.json) and [frozen_source_sha256.json](frozen_source_sha256.json) | Registered serving settings and a byte-level receipt for the shared MiniRAG source snapshot. |
| [dev.csv](st/inputs/dev.csv) and [test.csv](st/inputs/test.csv) | Byte-identical copies of the frozen questions. |

## Checks run

```powershell
$phiPython = 'E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $phiPython -B -m reproduce.mhr_phi.test_profile
& $phiPython -B -m reproduce.mhr_phi.launch reproduce.mhr_phi.c0 contexts --split dev --seed 101 --pilot-test
```

- **Profile tests passed.** Five tests imported 33 modules, checked 34 module-level artifact aliases identified by AST, verified the Qwen write and dotenv guards, and ran two real offline children through `subprocess.run` and `subprocess.Popen`. Mocked responses exercised the serving gates and the strict Phi response-model check.
- **Pilot C0 phase B passed.** Five questions produced five `ok` contexts and five retrieval-log records. All three workers exited 0, the cache-only flag was enabled, and the run made zero LLM calls. The pilot manifest's 10 file hashes matched and 0 of 69 chunks lacked graph coverage. The retrieval output hashes matched their receipt. [Context summary](_offline_test/c0_pilot__hsl3efg/C0_s101/contexts_summary.json)
- **Temporary fixture verified.** The original five Qwen cache lines were copied byte for byte. The pilot graph supplies 72 entity types, and all five cache entries needed a new prompt hash. The synthetic entries preserved the original keyword response text and query IDs. All pilot outputs stayed under `_offline_test/`; the production keyword paths remained absent. [Fixture receipt](_offline_test/c0_pilot__hsl3efg/fixture.json)
- **Question copies verified.** Dev has 400 rows and SHA-256 `4633ef777b747ef0fb8cca86457454eed64680432d899a3d5410afce2b63ef77`. Test has 2,156 rows and SHA-256 `e37a4599e8485b5b76f6b60157648b60564f51ee7eceaeb6798f7dbd523d069d`.
- **CE secondary scripts compiled with Phi paths.** Live judging and artifact-dependent reporting remain operator steps.

The initial pilot attempt stopped when the offline guard rejected the Windows asyncio wakeup socket. The final guard permits only CPython's loopback socket-pair setup for that operation. External socket connections and DNS remain blocked during offline steps. The initial attempt files were retained.

## Remaining assumptions and blockers

- Gate I requires the complete 609-article Phi index. Its manifest was absent at verification time. The pilot covered dev phase B. Full-index and test-cohort retrieval, generation, and judging await the production run.
- The unchanged Modal app loads Hugging Face main. Public preflight checks both Phi model names and vLLM 0.11.0. The operator needs server-side evidence for the registered weights commit when the response omits it.
- The CE and SET protocols explicitly fix the pool at `native@all`. A smaller cut selected by Phi's registered F dev rule requires resolving the CE and SET pool definition before those rows proceed.
- The shared b640765 source, temporal slide, embedding receipt, corpus, gold labels, cross-encoder revision, judge, and seeds remain the registered controls. The in-memory profile routes Qwen-specific paths, serving specs, report labels, and archive dependencies to Phi. The historical shell runners and migration metadata remain references for the original experiment. [Qwen assumptions in the runbook](../../reproduce/mhr_phi/RUNBOOK.md#qwen-assumptions-and-handling)

## Keyword guard

`reproduce/mhr_phi/kwguard.py` checks the frozen keyword reply at MiniRAG's cache lookup, before retrieval. The Phi loader in `profile.py` installs the check and adds the guard fields to the existing ladder writers in memory. `c0.py` records the failure text and count in C0 outputs. The frozen answer runner's `status != "ok"` branch emits `PROMPTS["fail_response"]` before constructing an answer prompt or invoking the LLM. ST inherits the guarded C0 record, F captures an empty pool, and CE and SET carry the failure into their own contexts and summaries.

### Offline checks

- `python -B -m reproduce.mhr_phi.test_kwguard` passed all four tests. Phi dev s101 and test s101 had zero bad replies across 400 and 2,156 cached records, respectively, and emitted zero events through the actual cache lookup. Phi test s202 had one bad reply across 2,156 records, `mhr-0315`. The Qwen dev cache and all three Qwen test caches had zero bad replies.
- Full-index C0 and F pool tests on `mhr-0315` and three controls verified that the guard bypassed both retrieval adapters. C0 stored `early_exit` with `PROMPTS["fail_response"]` and F stored `early_exit` with an empty pool. The three control contexts matched both the unguarded runs and the prior s202 worker records byte for byte. All control pool fields matched the unguarded runs apart from elapsed time. [Test artifacts](_offline_test/kwguard_tcrq2zaq/)
- Smoke checks covered the frozen dev context writer, ST worker, and F context worker. Fixture checks verified CE and SET inheritance, empty chunks, and per-cohort counts. `python -B -m reproduce.mhr_phi.test_profile` passed all five tests. [Worker smoke artifacts](_offline_test/kwguard_path_checks_934ql4z7/) [CE and SET fixture artifacts](_offline_test/kwguard_inheritance_bb56ojrq/)
- A separate trial with CPU math thread limits changed the serialized floating-point scores in a control context. The passing full-index tests and the production resume used the original thread settings.

### Production C0 test seed 202

```powershell
$env:PYTHONUTF8 = '1'
$phiPython = 'E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe'
& $phiPython -B -m reproduce.mhr_phi.launch reproduce.mhr_phi.c0 contexts --split test --seed 202
```

The original resume logic accepted the checkpoint. Shards started with 719, 89, and 718 contexts and ended with 719, 719, and 718. All 1,526 saved contexts and their retrieval-log prefixes verified byte for byte. The worker added 630 records, and all three workers exited zero.

| Cohort | Contexts | `ok` | `early_exit` | Keyword guard count |
| --- | ---: | ---: | ---: | ---: |
| Dev s101 | 400 | 400 | 0 | 0 |
| Test s101 | 2,156 | 2,156 | 0 | 0 |
| Test s202 | 2,156 | 2,155 | 1 | 1 |

The s202 merge contains every question exactly once in input order. `mhr-0315` has the failure text, one guard event, and no retrieval record. All 2,155 other queries have retrieval records and `ok` contexts. Both merged file hashes match the summary. The frozen keyword cache passed its original hash check, all 16 frozen source hashes and all 10 index file hashes verified, and the run made zero LLM calls. The s202 answer phase was left unrun.

[Context summary](c0/archive/qa/s202/C0_s202/contexts_summary.json) ? [Production guard event](kwguard_events.jsonl) ? [Verification receipt](_offline_test/kwguard_s202_verified.json)
