# Patches to the Lexsi libraries

These are bug fixes to the four Lexsi Labs libraries at the commits we pin. They were made on 9 Nov 2026 for this hackathon entry.
- **Installing them:** `scripts/setup_lexsi.sh` applies each patch to a fresh clone at its pinned commit and installs it in editable mode.
- **In the team folder:** the same changes sit on a `safety-pencil-patches` branch in each clone under `repos/`.

**Licence.** Each patched copy is a modified version of Lexsi Labs software under LSAL. We publish only these diffs, and we send no pull requests (SafeTune §6).

**Testing.** Every patch was checked against a clean clone with `git apply --check`. AuditKIT and CuratorKIT ran their test suites before and after: no regressions, plus 12 and 9 new tests. SafeTune and CircuitKIT compile, and each change was checked against their tests' expectations, but those suites need torch and were not run.

## CircuitKIT (`227fafd`)

| Change | Where | Why |
| --- | --- | --- |
| MLP node scores take the gradient at the MLP **output**, not its input | `src/circuitkit/backends/eap/attribute_node.py:170` | Node attribution is (activation change at the node's output) · (gradient at that output). Heads already did this. Upstream EAP-IG uses the output hook, and on a toy parallel-block model the old version agreed in sign with the true patching effect only about half the time. The one line feeds every gradient-based node method (EAP, EAP-IG, clean-corrupted, RelP, EAP-GP, AtP+GD, PEAP); edge-level code is unchanged. **MLP scores from before this patch are wrong; head scores are unaffected** |
| The parity test's model can be set with `CIRCUITKIT_PARITY_MODEL` | `tests/backends/tl_compat/test_cohere_parity.py:39` | So the TL-vs-HF parity check runs on tiny-aya-global, not just base |

## SafeTune (`1d824f0`)

| Change | Where | Why |
| --- | --- | --- |
| `RefusalDirectionTrainer(..., config=RefusalDirectionConfig(...))` | `src/safetune/runner/steer/_activation.py` | The trainer never passed a config, so `pick_layer` couldn't be set |
| C-ΔΘ demo uses `positive=aligned, negative=base` | `examples/demos/compare_recover.py:66` | It applied base − aligned, the reverse of the safety direction |
| C-ΔΘ difference and update computed in fp32 | `src/safetune/interventions/recover/ctheta.py` | The difference was taken in the parameter dtype (bf16) |
| LoX updates a tied tensor once | `src/safetune/interventions/recover/lox.py` | With tied embeddings, a later alias could overwrite the extrapolated weight |
| ReSta raises on a base/aligned shape mismatch | `src/safetune/core/optim/resta.py` | It broadcast silently or failed unclearly |
| SafeMERGE docstring describes cosine similarity | `src/safetune/runner/recover/_layer.py` | It said "Fisher-weighted"; the code uses no Fisher |
| `SafeDeltaTrainer(..., hessian_inputs=...)` | `src/safetune/runner/recover/_layer.py` | The trainer never passed curvature, so it always ran with H = I. We feed it our A factors as a curvature-aware baseline |

## AuditKIT (`9da5838`)

| Change | Where | Why |
| --- | --- | --- |
| `AUDITKIT_DISABLE_RUN_CACHE=1` turns off the run cache (read and write) | `src/auditkit/cache.py`, `runner.py`, `lmeval_engine.py` | Every run wrote full JSON, raw completions included, to `~/.cache/auditkit/runs`, with no way to stop it. Our rule is no raw harmful completions on disk |
| A local checkpoint's fingerprint includes its weight files' names, sizes and modification times | `src/auditkit/model/__init__.py` | A checkpoint retrained at the same path silently reused the old cached result. Hub ids are unchanged |
| `AutoModel.resolve(..., name=...)` works | `src/auditkit/model/__init__.py` | It raised a duplicate-keyword `TypeError` |
| Docs use `p.prompt` | `docs/guard_models.md:233` | `Prediction` has no `input` field |

Still writing raw outputs when called explicitly: `RunResult.save`, `ExperimentDB.save` (via `experiment_name=`) and `--output` flags. Don't call them on harmful prompts.

## CuratorKIT (`cee1aed`)

| Change | Where | Why |
| --- | --- | --- |
| New `prompts_meta` exporter: JSONL rows `{id, prompt, task_type, source_uri, metadata}` | `curatorkit/exporters/prompts_meta.py`, registered in `exporters/__init__.py`; allowed for every task type (`compatibility.py`) and in the YAML config (`config.py`) | Prompt-only sets could only use `ppo`, which drops id, language, harm category and source. It is written only when listed in `export_formats` |
| The PDF sentence chunker also splits after "।" and "॥" | `curatorkit/connectors/pdf.py:34` | Hindi text never split; English behaviour is unchanged |
| OCR language is configurable: `pdf_ocr_lang` in `CuratorConfig`, `ocr_lang` on the PDF reader | `curatorkit/connectors/pdf.py`, `curator.py` | It was hard-coded to English. Check MinerU's code for Hindi (likely `"devanagari"`) before using it |
| Provenance records the dataset split | `curatorkit/manifest.py`, `curator.py` | Only the name (and subset) were recorded |
