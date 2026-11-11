# Experiment log

One line per run, newest last. Commit the code, and this log with it, before a run (results record the git
hash, with `-dirty` if any tracked file had uncommitted changes) and commit the result files right after it.
From 12 Nov every Hugging Face Job also records itself in the run ledger (`runs/<run id>/` in the private runs
repo: record, log, and a copy of each result it wrote; see `jobs/ledger.py`), and `jobs/report.py` turns the
ledger and the results into one page, `outputs/summary.md`. This file stays the place for notes and decisions. Decisions go in the team's
`03-team/DECISIONS.md`; changes to anything pre-registered also go in `PREREGISTRATION.md`'s deviation log.

| Date (IST) | Who | Code commit | Command | Result files | Outcome / notes |
| --- | --- | --- | --- | --- | --- |
| 11 Nov 20:50 | Tanay (Claude) | 9e11b61 | `jobs/launch.py calibrate` (HF Job 6af4878dab34849891ae3945, l4x1, ~$1) | — | ERROR after ~12 min (~$0.20): pytest and access OK (all 4 models incl. Llama Guard); harmful en 388/97/485, hi 362/91/453 (factor/dev/test, stratified); 01_build_task_t1 stopped: only 889 usable Hindi rows in aya_dataset (1153 raw), need 4300. Follow-on jobs not started. |
| 11 Nov 21:29 | Tanay (Claude) | 81b27e5 | `jobs/launch.py calibrate`, relaunch 1/3 after the T1 top-up fix (HF Job 6af490949737d9a9e09312ca, l4x1, ~$1) | data/processed, outputs/results (runs repo), configs/model_facts.json | COMPLETED (queued 42 min for an L4, ran ~7 min, ~$0.15). T1 heldout 300 / train_4000 / train_1000 (889 aya_dataset + top-up); XSTest en dev 50 / test 200; leak check PASS; model facts as expected (parallel block, no v bias/norm, 1/w null, h // 4, d_mlp 11008); run A 3984 rows (16 too long), 6.4 s/step × 498 steps ≈ 0.89 h. Results stamped 81b27e5-dirty (only EXPERIMENTS.md was modified). |
| 11 Nov 22:17 | chained from calibrate | 81b27e5 | hinglish (6af49c1b9737d9a9e0931d7e, cpu-basic); finetune A (6af49c1cab34849891ae4709), C (6af49c1c9737d9a9e0931d80), D (6af49c1c9737d9a9e0931d83), l4x1, ~$1 each | — | started. hinglish CANCELED after ~25 min (~$0.02): every router call got HTTP 401 because CuratorKIT's LiteLLMBackend sends api_key="nokey" when given an api_base and no key, so OPENAI_API_KEY was ignored; the script has no zero-output check and would have started evaluate theta0 on empty Hinglish data. Fix: make_llm passes $OPENAI_API_KEY explicitly (uncommitted). Finetunes unaffected. |
| 11 Nov 23:22 | Tanay (Claude) | c1d90e0 | `jobs/launch.py hinglish` (HF Job 6af4ab47ab34849891ae5984, cpu-basic) | data/processed/harmful/hinglish, data/processed/xstest/{hi,hinglish}, outputs/review | COMPLETED in ~15 min (~$0.01). Hinglish harmful kept factor 510/750, dev 134/188, test 635/938 (rejects: length ratio ~16%, the converting model's refusals ~15%); XSTest hi 250/250, Hinglish 246/250; leak check PASS. Started evaluate theta0 (6af4af71ab34849891ae5fbf, l4x1). |
| 11 Nov 22:17–23:40 | chained from calibrate | 81b27e5 | finetune A, C, D (above) | outputs/ckpt/{A,C,D} | Training done: A 498 steps (held-out loss 1.69–1.70), C 124 steps, D 126 steps; each then probe-evaluates its 4 adapters with the old evaluate.sh (results named probe_{A,C,D}_frac*, renamed once). C and D were not cancelled: they had an L4 and were training by the time a switch was considered. |

Notes:
- T1 mix (11 Nov): about 589 human-written aya_dataset rows remain after the 300 held-out rows (all human), so
  train_1000 (runs C, D) is ~59% human-written and train_4000 (run A) ~15%; the rest is Aya Collection top-up.
  Run A vs run C therefore differs in data mix as well as size. 16 of 4000 rows skipped as too long is negligible.
- Probe result names (11 Nov): run A's job was launched with the old evaluate.sh, which writes probe_A_frac*.json;
  06_repair.py forecast reads probe_run{R}_frac{F}. evaluate.sh is fixed for C, D and later runs; run A's four
  files are renamed once in the runs repo.
