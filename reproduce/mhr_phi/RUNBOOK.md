# Phi MultiHop-RAG ladder

Run from the MiniRAG-Improvement repository root in PowerShell. The commands in the operator sequence include paid generation, judging, and cross-encoder scoring. The registered inputs, gates, and analysis are in [PROTOCOL.md](../../review/mhr_phi_20261118/PROTOCOL.md).

## Offline checks

```powershell
$phiPython = 'E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $phiPython -B -m reproduce.mhr_phi.test_profile
& $phiPython -B -m reproduce.mhr_phi.launch reproduce.mhr_phi.c0 contexts --split dev --seed 101 --pilot-test
```

The pilot command creates a fresh directory in `review/mhr_phi_20261118/_offline_test/`. It copies five original keyword records and re-keys the synthetic cache by each pilot prompt's SHA-256 because the graph's sorted `TYPE_POOL` differs. The synthetic records keep the original response text and question IDs. Both fixtures and all worker outputs stay inside that test directory.

## Operator setup

Verify gate I on the completed `MultiHop-RAG-phi-ropefix609/` index. C0 checks the manifest, all frozen file hashes, the 609-document scope, the registered extraction settings, and the overall 5% zero-entity threshold. Confirm the deployed LongRoPE app loaded the registered Hugging Face commit `2fe192450127e6a83f7441aef6e3ca586c338b77` from its server-side cache or deployment records. Record that evidence with the run.

```powershell
$phiPython = 'E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MINIRAG_DOTENV = 'E:/Project/MiniRAG research/MiniRAG/.env'
$env:PHI_SLM_URL = '<HTTPS URL of the deployed minirag-slm-phi-ropefix endpoint>'
$phiSeeds = 101, 202, 303

function Invoke-Phi {
    & $phiPython -B -m reproduce.mhr_phi.launch @args
    if ($LASTEXITCODE -ne 0) { throw "Phi command failed with exit code $LASTEXITCODE" }
}
```

The launcher loads credentials with `override=False` for online steps and passes them to its children in the environment. The Phi URL takes precedence for `MINIRAG_SLM_URL`. Offline stages work independently of the credential file. Record Modal spend per stage and workspace switches in `review/mhr_phi_20261118/DEVIATIONS.md` as specified by the protocol.

## C0

```powershell
Invoke-Phi reproduce.mhr_phi.c0 execute
```

The controller runs A → B → C for dev seed 101, then for test seeds 101, 202, and 303 in order. Rerun `execute` after an interruption. The same steps can be run individually for a cohort.

```powershell
Invoke-Phi reproduce.mhr_phi.c0 keywords --split dev --seed 101
Invoke-Phi reproduce.mhr_phi.c0 contexts --split dev --seed 101
Invoke-Phi reproduce.mhr_phi.c0 answers --split dev --seed 101
```

Use `--split test --seed <101|202|303>` for the test cohorts. C0 logs its current stage to `c0/status.json`. The dev run is `c0/dev/C0_s101/` and each test run is `c0/archive/qa/s<seed>/C0_s<seed>/`. Phase A requests are traced in each cohort's `keywords/requests.jsonl`. Phase C traces are in the cohort's `requests.jsonl`.

The dev context phase uses the frozen `phased_qa.context_phase`. The test phase runs the C0 half of the frozen Phase 3 shared prefix/suffix worker. Both use three round-robin workers with `PYTHONHASHSEED=13`, local embeddings, `QueryParam(mode='mini', only_need_context=True)`, default top-k 60 and a 4,000-token Sources budget. The frozen environment enables the answer-type fix and Sources truncation, disables the path-to-chunk fix, clears chunk cut and fusion, and applies the 1,024-token generation cap and per-request seed. Context workers use `MINIRAG_KW_CACHE_ONLY=1` and a rejecting LLM. Generation uses the frozen keyword and answer prompt builders, `QueryParam().response_type`, and concurrency 32.

## +ST

```powershell
Invoke-Phi reproduce.mhr_st.run prepare
Invoke-Phi reproduce.mhr_st.run preflight
Invoke-Phi reproduce.mhr_st.run contexts --split dev --seed 101 --limit 20 --workers 2
Invoke-Phi reproduce.mhr_st.run execute --workers 3
Invoke-Phi reproduce.mhr_st.finish
```

The dev20 smoke must pass the frozen C0 replay gate. `execute` produces test contexts and answers. `finish` judges C0, judges the changed ST answers, and composes ST verdicts with same-seed C0 reuse.

## +F

```powershell
Invoke-Phi reproduce.mhr_f.pool run --split dev --seed 101 --workers 3
Invoke-Phi reproduce.mhr_f.study --seed 101
Invoke-Phi reproduce.mhr_f.run freeze
foreach ($seed in $phiSeeds) {
    Invoke-Phi reproduce.mhr_f.pool run --split test --seed $seed --workers 3
}
Invoke-Phi reproduce.mhr_f.run execute --workers 3
Invoke-Phi reproduce.mhr_f.run finish
```

The dev rule chooses the smallest registered native cut within `ceil(2%)` of the uncut full-evidence count, with `native@all` as the fallback. The frozen scorer uses the resulting Phi dev pool and the registered 18 features.

## +CE

The scorer rents a Modal A10G and compares its logits against a local CPU probe. Use the pinned `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Check the pool eligibility described under **Blocked** before scoring.

```powershell
Invoke-Phi reproduce.mhr_ce.scores --split dev
Invoke-Phi reproduce.mhr_ce.study --seed 101
Invoke-Phi reproduce.mhr_ce.run freeze
Invoke-Phi reproduce.mhr_ce.scores --split test
foreach ($seed in $phiSeeds) {
    Invoke-Phi reproduce.mhr_ce.run contexts --seed $seed
}
Invoke-Phi reproduce.mhr_ce.run execute
Invoke-Phi reproduce.mhr_ce.run finish
Invoke-Phi reproduce.mhr_phi.secondary ce-drift
Invoke-Phi reproduce.mhr_phi.secondary ce-metrics
```

Run `ce-drift` on the CE judging date. It re-judges the registered 200 seed-101 F answers under Phi. `ce-metrics` prints the registered entity-substring and Fleiss κ checks from Phi outputs.

## +set

```powershell
Invoke-Phi reproduce.mhr_set.test_select
Invoke-Phi reproduce.mhr_set.study
Invoke-Phi reproduce.mhr_set.run freeze
foreach ($seed in $phiSeeds) {
    Invoke-Phi reproduce.mhr_set.run contexts --seed $seed
}
Invoke-Phi reproduce.mhr_set.run execute
Invoke-Phi reproduce.mhr_set.run finish
```

The existing SET judge code applies the November 18 amendment from its first attempt. A slot with at least five `parse_failed` cycles is marked `unparsed` and contributes no accurate vote. SET finish also runs the registered CE seed-101 drift sample, entity metric, and Fleiss κ checks.

## Qwen assumptions and handling

| Location | Phi handling |
|---|---|
| `mhr_st.common`, `mhr_f.pool`, `mhr_ce.offline`, `mhr_ce.scores`, `mhr_set.offline` | The in-memory loader redirects the index, archive, and four output roots before module code executes. Derived `MODEL`, `QUESTIONS`, `PARAMS`, `SCORES`, and pool paths follow those roots. The AST test verifies every by-name copy of a changed binding. |
| `mhr_st.pipeline.reference`, `mhr_f.pool.DEV_C0_SRC`, and `archive_c0` lookups in F, CE, and SET | The dev reference points to the newly built Phi dev C0. All test baseline lookups, including report evidence-recall inputs, use the Phi archive layout. |
| `mhr_st.run.prepare`, `mhr_st.pipeline.execute`, and `mhr_st.finish.amendment` | Preparation verifies the shared source, Phi index, question copies, embedding receipt, and each completed Phi C0 cohort. The old archive-copy code and Qwen migration amendment are bypassed. Judging runs in `finish` to apply the registered reuse policy. |
| `common.setup`, `offline.setup`, `adapter.load_temporal`, `mhr_ce.local_runtime`, and embedding receipt references | The MiniRAG b640765 snapshot, its transport patch, temporal slide 1297b2e, and the embedding receipt stay shared read-only. The local embedding files must match the receipt. `dataset/`, question bytes, and gold labels remain the registered inputs. |
| `mhr_st.run.preflight`, ST and F serving gates, and CE/SET delegates | The launcher supplies the environment and suppresses implicit dotenv loads. Serving checks use the Phi JSON and require both Phi names and vLLM 0.11.0. Generation traces require the canonical Phi response model. |
| Text reports and manifests | The banner is `Phi-3.5-mini-instruct (LongRoPE fix)` and the recorded SLM is `microsoft/Phi-3.5-mini-instruct`. SET reports its registered secondary comparison. Paired decisions are within each Phi seed. |
| Frozen `phased_qa` and older `qa_run`, `run_mhr_qa.sh`, `shard_mhr_qa.py`, and `run_phased_dev.sh` entry points | The profiled command uses the frozen phase functions and replaces the Qwen default index, automatic credential loading, and permissive trace filter in memory. The older shell runners and archived Qwen controllers are provenance for the settings. |
| Archived CE secondary scripts | The Phi secondary command compiles the registered scripts with artifact paths rebound in memory. Historical standalone diagnostic scripts and their saved results remain historical. |
| Fixed checks and literals | Seeds 101/202/303, hash/fold/sample seed 13, judge `gemini-3.5-flash-lite`, CE revision, source commit, retrieval settings, and sample sizes are registered shared controls. The F and CE replay checks use Phi references and hashes. The CE and SET `native@all` requirement remains a preregistered gate. |

## Blocked

- **Gate I.** Production C0 needs the completed full Phi index. The pilot supports the offline check.
- **Served revision.** The unchanged Modal app does not pin `--revision`. The ordinary `/version` and `/v1/models` checks establish the vLLM version and served names. They establish the weights revision only if the endpoint supplies a revision field. Use deployment or server-side cache evidence for the revision when that field is absent.
- **CE and SET pool eligibility.** The original CE and SET protocols explicitly specify `native@all`. If the registered Phi F rule chooses a smaller cut, finish F and pause CE and SET until the pool choice is resolved in the experiment record. The CE data check and SET study enforce this gate.
