# How the Safety Pencil is built on the four Lexsi repos

Read on 9 Nov 2026 from the source code, not the READMEs, at these commits. Clones at exactly these commits are in `repos/` in the team folder, for reading only:

| Repo | Commit (7 Nov) | Version in repo | On PyPI |
| --- | --- | --- | --- |
| SafeTune | `1d824f0` | 0.1.12 | 0.1.13 |
| CircuitKIT | `227fafd` | 0.2.1 | 0.2.1 |
| AuditKIT | `9da5838` | 1.1.6 | 1.1.6 |
| CuratorKIT | `cee1aed` | 0.1.21 | 0.1.23 |

Install from these commits (`pip install git+https://github.com/Lexsi-Labs/<repo>@<commit>`) so the code matches what is described here. Paths below are relative to each repo, so `src/safetune/...` means a file in SafeTune.

---

## The short answer

The repos give us data handling, baselines, circuit discovery and evaluation. The method itself is ours, because **none of the four repos computes a Fisher, a Hessian or K-FAC factors.** SafeTune's Safe Delta accepts a Hessian but never builds one.

| Stage | Repo | What the repo gives | What we write |
| --- | --- | --- | --- |
| 1. Data | CuratorKIT | Hugging Face loading, exact and MinHash dedup, seeded splits, Hindi QA from PDFs, provenance files | Hinglish conversion step, script/refusal gates, a metadata-keeping exporter, dataset cards |
| 2. Drift (LoRA fine-tune) | SafeTune (helpers) | `lora_wrap`, preamble-aware `tokenize_dataset`, BeaverTails contamination pairs | A short PEFT training loop that **keeps the adapter** |
| 3. Curvature factors | none | nothing | Forward and output-gradient hooks on the HF model, with sampled labels |
| 4. Pencil maths | none | nothing | `src/pencil/` (gen_eig, repair, leverage, forecast), already specified and tested |
| 5. Repair as a SafeTune method | SafeTune | A recover registry and base class | `PencilTrainer`, registered as `pencil` |
| 6. Baselines | SafeTune | ReSta, LoX, C-ΔΘ, Safe LoRA, SafeMERGE, task arithmetic, Pre/Post merge, WiSE-FT, SOMF, Safe Delta, RefusalDirection | RegMean, diagonal Fisher merge, AlignMerge-style merge, standard Fisher merge; feed our A factors to Safe Delta |
| 7. Locate cross-check | CircuitKIT | Tiny Aya support, EAP-IG node scores per head and per MLP, faithfulness pillars | A custom refusal task, length-matched pairs, a random-layer control |
| 8. Evaluation | AuditKIT | Batched generation, Llama Guard 3 judge, rubric judge, per-sample labels | Statistics (Wilson, McNemar, Holm, κ intervals), batched judging, sanitised exports |
| 9. Publishing | all three write provenance files | — | `huggingface_hub.upload_folder`, our own dataset and model cards, collections |

---

## 0. Environments and licences

**Two virtual environments**, because CircuitKIT pins TransformerLens exactly.

| Environment | Python | Install | transformers |
| --- | --- | --- | --- |
| `.venv` (main) | 3.12 (SafeTune needs ≥3.12) | `safetune`, `auditkit[transformers,interop]`, `curatorkit[hf,generation]`, `peft`, `scipy` | **pin `5.21.1`**: AuditKIT needs `>=5.15,<5.22`, SafeTune `>=5.15,<6` |
| `.venv-circuit` | 3.10+ | `circuitkit` (pulls `transformer-lens==3.8.0`) | `>=5.15,<6`. Always `import circuitkit` before `transformer_lens`, or it raises ImportError |

- Add `curatorkit[pdf]` (MinerU) only on the machine that parses PDFs, and `curatorkit[embedding]` only if you use embedding dedup.
- On Colab, run `pip uninstall -y torchao` before importing SafeTune (its README; the cause isn't documented).
- All four Tiny Aya Hub ids are gated. Accept the licence and put an `hf_...` token in `.env`. CircuitKIT checks the prefix.

**Licences.** All four repos use LSAL (non-commercial, research use).
- **§1A of SafeTune, CircuitKIT and AuditKIT.** An *organisation* using them on its own models for evaluation or safety work needs written permission, or must send an acknowledgement to support@lexsi.ai. If the team enters as individuals, §1 (personal and academic research) applies. Decide which before submitting.
- **SafeTune §4.** Never deploy a drifted checkpoint, or any version that hasn't been repaired and re-evaluated. Our model cards already say "research artifact: unrepaired, do not deploy".
- **SafeTune §6.** Any contribution sent to Lexsi (a pull request) gives them perpetual commercial rights to it. That affects the "open a SafeTune PR with the pencil repair" item in the checklist. A public plugin in our own repo gets the same visibility without that clause.

---

## 1. What Tiny Aya's code fixes in our design

Checked in `transformers/models/cohere2/modeling_cohere2.py` (transformers 5.22.2):

| Fact | Where | Consequence for us |
| --- | --- | --- |
| Parallel block: `h + attn(LN(h)) + mlp(LN(h))` from one `input_layernorm` | `Cohere2DecoderLayer.forward` | q, k, v, gate and up read the same input, so **one shared A factor per layer** |
| `Cohere2LayerNorm(bias=False)`, mean-subtracting | `Cohere2LayerNorm` | That shared A has the exact null vector 1/w (w = LayerNorm weight). Range restriction is mandatory |
| GQA, 16 query heads over 4 KV heads, `repeat_kv` | `eager_attention_forward` | Query head h reads KV head h // 4. The V/O gauge is a 128×128 matrix per KV group |
| RoPE only on sliding-window layers; every 4th layer is full attention with no positional encoding | `Cohere2Attention.forward`, `layer_types` | The V/O gauge is unaffected (values are never rotated). A Q/K gauge would have to commute with RoPE, so stick to V/O |
| `logits * logit_scale`; tied embeddings | `Cohere2ForCausalLM` | Leave `lm_head` and the embeddings out of the repair. LoX mishandled tied embeddings (fixed in our patch) (`src/safetune/interventions/recover/lox.py:109-148`) |
| No q/k norm, no attention bias by default | `Cohere2Attention.__init__` | Prompt 0 still checks `attention_bias` in the real config |

**The chat template matters for safety.** Tiny Aya's template always injects a system preamble that begins "You are in contextual safety mode…", then a default preamble, about 366 tokens in all (AuditKIT `tests/fixtures/chat_templates/tiny_aya_default.jinja:1-16`). A system prompt is added as a "Developer Preamble" below it and does not replace it. So part of the model's refusal behaviour comes from the template. **Use the identical template for fine-tuning, factor passes, discovery and evaluation,** and pin the tokenizer files so a PEFT adapter directory can't bring a different template.

---

## 2. Stage by stage

### Stage 1 — Data (CuratorKIT)

**What it gives.**
- **High level:** `Curator(CuratorConfig(...)).run()` (`curatorkit/curator.py:126,679`). The steps run in a fixed order: readers → schema gate → exact/MinHash dedup → cleaning → hygiene → generation → gates → exporters. It then writes `manifest.json`, `lexsi_provenance.json`, `rejected.jsonl`, `checksums.txt` and a dataset card.
- **Low level:** `Pipeline(steps, output_dir)` with `.run()` or `await .run_async()` (`curatorkit/pipeline.py:80-86`), for custom steps. It writes no provenance files, so call `ProvenanceManifest(...).write()` and `DatasetCardGenerator().generate(...)` yourself (`curatorkit/manifest.py`).
- **Splits:** `output_split={"factor": .4, "dev": .1, "test": .5}` with `output_split_seed`. Dedup runs before splitting, so no duplicate crosses splits within one run (`curator.py:1031-1043` vs `1705-1749`).
- **Toxicity filtering is off by default** (`toxicity_gate: bool = False`, `curator.py:461`). Leave it off for the harmful sets.
- **QA from PDFs:** `generation_task="qa"` gives question/context/answer rows; a custom template must contain `{context}` and `{num_questions}` (`curatorkit/generators/qa_generator.py:115-133`). `HallucinationGate` checks grounding (`curatorkit/gates/hallucination.py:98-121`).
- **LLM backend:** `LiteLLMBackend("openai/CohereLabs/aya-expanse-8b", api_base=URL)` works with a local vLLM server (`curatorkit/llm/litellm.py:82-133`).

**What we write.**
- **Hinglish conversion.** CuratorKIT has no translation or transliteration step, and no language ID. Subclass `BaseGenerationTask` (implement `_build_messages` and `_parse_response`, `curatorkit/generators/base.py:81-87,164-172`) and run it through `Pipeline`. Reject outputs that still contain Devanagari, and check that the converter didn't refuse or soften the harmful prompt.
- **Prompt-only exports: use `export_formats=["prompts_meta"]`.** The built-in `ppo` exporter writes only `{"prompt"}` and drops id, language, harm category and split (`curatorkit/exporters/compatibility.py:49`, `ppo.py:47`). Our patch adds a `prompts_meta` exporter that keeps them (`patches/README.md`).
- **Dataset cards.** The generated README front matter has only `tags: [curatorkit]`, and Hugging Face sources show as "Unknown" (`manifest.py:153-160,351-362`). Write licence, language and content-warning fields ourselves.

**Gotchas.**
- **Short prompts get rejected:** the default `min_tokens=10` counts whitespace words (`curator.py:213`), so set `min_tokens=1`.
- **Ids change every run:** each run assigns a fresh uuid4 unless the row has an `id`. Add a deterministic id, a hash of language + prompt, with `preprocessing_fn`.
- **Split membership isn't stored on the rows,** only in the folder layout. Run Curator once per language, and make Hinglish from the per-split files, never by re-splitting.
- **aya_dataset columns aren't recognised.** It needs `field_mapping={"inputs": "instruction", "targets": "output"}`, or every row is rejected. Pre-filter to Hindi with `datasets` first.
- **aya_redteaming:** languages are splits of the default config (`split="hindi"`) according to the card's YAML, but its code example uses configs. Check once.
- **PDFs:** with our patch the sentence chunker splits after "।" and "॥", and the OCR language is configurable (`pdf_ocr_lang` in `CuratorConfig`). Check MinerU's code for Hindi (likely `"devanagari"`), and spot-check the extracted text.
- **Dedup against the fine-tuning set:** `EmbeddingDeduplicator` defaults to an English embedding model. Pass a multilingual one, and never put Hindi and Hinglish twins in one index, or the Hinglish copies get dropped.

### Stage 2 — Drift: the LoRA fine-tune (our loop, SafeTune helpers)

SafeTune's `harden.PlainSFTTrainer` merges the LoRA and **discards the adapter** (`src/safetune/runner/harden/_base.py:327-340`). It also uses batch size 4 with no gradient checkpointing, which can run out of memory once the preamble is counted. Our repair needs the fp32 adapter (ΔW = scaling · B @ A), so we write a short PEFT loop and reuse two helpers:

```python
from safetune.runner.utils.model_utils import lora_wrap            # model_utils.py:106
from safetune.runner.utils.data_utils import tokenize_dataset, harden_contamination_pairs
model = lora_wrap(model, r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
train = tokenize_dataset(rows, tok)   # data_utils.py:236; chat template applied, loss on the last assistant turn,
                                      # max_len sized to cover the ~366-token preamble
contam, _tar, _aux = harden_contamination_pairs(n)   # data_utils.py:494; three lists of (prompt, response)
```

- `tokenize_dataset` accepts `messages`, ShareGPT, Alpaca or prompt/response rows (`data_utils.py:152-293`). It pads every row to `max_length`.
- For the 2% contamination variant, take the first list from `harden_contamination_pairs`. It is BeaverTails `30k_train`, English only.
- Save the adapter with `model.save_pretrained(...)`; never only the merged weights.

### Stage 3 — Curvature factors (ours)

No repo helps here, and CircuitKIT's TransformerLens port is the wrong tool (it keeps separate embedding copies and is built for discovery). Hook the **HF model** directly:

- **Inputs a:** a forward hook on each `Linear` captures its input. Thanks to the parallel block, compute the shared A once per layer from the `input_layernorm` output, then add one A for `o_proj` input and one for `down_proj` input.
- **Output gradients g:** register a tensor hook on each Linear's output inside its forward hook. Freeze every parameter and set `inputs_embeds.requires_grad_(True)` so no weight gradients are stored.
- **Sampled labels:** at each loss position sample a token from the model's own softmax, and back-propagate the sum of their log-probs once (unbiased G; `VERIFICATION.md` Lemma 3.6).
- **Normalisation:** A is a mean over the rows used; G is a sum over the same rows divided by the number of loss tokens.
- **Two row masks in one pass:** response positions, and all user-prompt + response positions excluding the fixed system preamble.

**Memory.** Per layer and per (data side, row set), the fp32 factors hold 5·2048² + 2·512² + 3·d_mlp² numbers:
- A: shared input and `o_proj` (2048 each), `down_proj` (d_mlp).
- G: q, o and down (2048 each), k and v (512 each), gate and up (d_mlp each).

d_mlp is `intermediate_size` in the gated config (Prompt 0 records it). For example, at d_mlp = 8192 that is about 0.9 GB per layer, so about 32 GB for all 36 layers, per side and row set. So:
- hook a chunk of layers per pass and write each layer's factors to disk;
- keep the top 512 pairs on MLP-width sides after the eigendecomposition;
- pick the row set on a few layers first, then run the full set with one.

### Stage 4 — Pencil maths (ours, already specified)

`src/pencil/{geneig,repair,leverage,invariance}.py` exactly as in `CLAUDE.md` and `PROMPTS.md`:
- range restriction and covariant damping;
- the subtraction form, plus a step ratio for the trust-region check;
- leverage, forecast, frontier.

`tests/test_math.py::pencil_repair_np` is the reference the torch code must match.

### Stage 5 — The repair as a SafeTune recover method

SafeTune's recover methods share one interface:
- `XTrainer(drifted_model, **kw).apply()` modifies the **merged** drifted model in place and matches weights by `state_dict` key (`src/safetune/runner/recover/_base.py:17-124`).
- `RECOVER_REGISTRY` and `register_recover(alias, class_name)` add new ones (`src/safetune/runner/_registry.py:52-104`).
- The CLI finds classes with `getattr(safetune.runner.recover, name)` (`src/safetune/cli.py:196`).

So the pencil plugs in next to every baseline:

```python
import torch, safetune.runner.recover as RR
from safetune.runner.recover._base import _RecoverBase
from safetune.runner._registry import register_recover
from pencil.repair import repair_layer            # ours; returns (W0 + dW', step_ratio)

class PencilTrainer(_RecoverBase):
    METHOD = "PencilTrainer"
    def __init__(self, model=None, *, aligned_model=None, kfac_path=None, mu=0.3, gamma=1e-3, **kw):
        super().__init__(model, **kw)
        self.aligned, self.mu, self.gamma = aligned_model, mu, gamma
        self.kfac = torch.load(kfac_path)          # {module_name: (A_h, A_t, G_h, G_t)}
        self.step_ratios = {}
    def apply(self, **kw):
        al = self.aligned.state_dict()
        with torch.no_grad():
            for name, lin in self.model.named_modules():
                if name not in self.kfac: continue
                W0 = al[name + ".weight"].to(lin.weight.device)
                dW = lin.weight.float() - W0.float()   # better: the fp32 adapter product, passed in directly
                W_new, self.step_ratios[name] = repair_layer(W0, dW, *self.kfac[name], self.mu, gamma=self.gamma)
                lin.weight.copy_(W_new)
        return self.model

RR.PencilTrainer = PencilTrainer       # makes the CLI lookup work
register_recover("pencil", "PencilTrainer")
```

`ΔW` from merged bf16 weights loses precision (`CLAUDE.md` rule 1). The real trainer should take the adapter path and rebuild ΔW in fp32. The skeleton above only shows the interface.

### Stage 6 — Baselines

**From SafeTune.** None of these needs data. All except WiSE-FT, Pre/Post merge and SOMF need the base model (`CohereLabs/tiny-aya-base`). Each loads two to three full models (about 6.7 GB each in bf16), so keep base and aligned on CPU with `load_model_cpu` (`src/safetune/runner/utils/model_utils.py:83`).

| Alias | Class and arguments (`src/safetune/runner/recover/`) | What it does | Notes |
| --- | --- | --- | --- |
| `resta` | `ReStaTrainer(base_model, aligned_model, alpha=1.0, dare=True, dare_drop_rate=0.3, device=None)` | adds α·DARE(aligned − base) | Use α ≈ 0.25 on Tiny Aya (`_layer.py:14-16`). The plain function `apply_resta` defaults to `dare=False` |
| `lox` | `LoXTrainer(base_model, aligned_model, rank=8, extrapolation_factor=0.3)` | adds c·SVD_k(aligned − base) | Restrict to `param_filter=["layers."]` via `safetune.recover.apply_lox`; tied-embedding issue above |
| `ctheta` | `CThetaTrainer(base_model, aligned_model, strength=1.0).apply(circuit_info=...)` | adds strength·(aligned − base) inside a circuit | Differences taken in fp32 with our patch (bf16 before). Circuit from `safetune.interpret.safety_circuit_info(...)` or a CircuitKIT JSON (`load_circuit_info_from_file`) |
| `safelora` | `SafeLoRATrainer(aligned_model, base_model, alpha=0.5, threshold=0.5)` | projects each `lora_B` where cosine ≤ threshold | Give it the **PEFT** model (paper version), then `merge_and_unload()` |
| `safemerge` | `SafeMergeTrainer(base_model, aligned_model, threshold=0.35, alpha=0.5)` | per-layer interpolation when cosine < τ | Its docstring says "Fisher-weighted"; the code uses no Fisher |
| `task-arithmetic` | `TaskArithmeticTrainer(base_model, aligned_model, alpha=1.0)` | adds α(aligned − base) | Tune α; α = 1 over-applies on a model that already contains the difference |
| `prepost` | `PrePostMergeTrainer(pre_model=..., alpha=0.5)` | (1−α)θ_drift + αθ_pre | With `pre_model` = aligned it equals WiSE-FT, so pass `tiny-aya-base` to make it distinct |
| `wiseft` | `WiseFTTrainer(aligned_model, alpha=0.5)` | αθ_aligned + (1−α)θ_drift | linear interpolation |
| `somf` | `SOMFTrainer(aligned_model, mask_threshold=0.9)` | magnitude mask | SafeTune's own audit marks it a variant, not the paper's method |
| Safe Delta | `SafeDeltaTrainer(..., hessian_inputs={param: H})` (our patch passes it through); or `safetune.recover.apply_safe_delta(..., hessian_inputs=...)` | Hessian-weighted safety delta | Without H it runs with H = I. **Feed it our A factors to get a curvature-aware baseline inside SafeTune** (`interventions/recover/safe_delta.py:86`) |

**Steering reference.** With our patch, `steer.RefusalDirectionTrainer(model, tok, config=RefusalDirectionConfig(pick_layer=24, target_layers=[24]))` passes the config through (before, it ignored it). Calling the functions directly also works:

```python
from safetune.interventions.steer.refusal_direction import (extract_refusal_direction,
    RefusalDirectionConfig, RefusalDirectionModel)
d, _, _ = extract_refusal_direction(model, tok, harmful, harmless,
                                    RefusalDirectionConfig(pick_layer=24, target_layers=[24]))
with RefusalDirectionModel(model, d, mode="steer", strength=s, layers=[24]):
    out = model.generate(...)
```

- The default strength of 20 is "far too strong" (SafeTune README).
- The extractor keeps full-sequence activations for every layer, about 15 GB of RAM per side at 256 prompts, so always pass `target_layers`.
- Its automatic layer search matches English refusal phrases, so fix `pick_layer` rather than searching.

**Ours** (same inputs as the pencil, so the comparison isolates the curvature model):
- **RegMean:** W = W₁ − ΔW·A_h(A_h + μA_t)⁺, which is the K-FAC merge with G = I.
- **Diagonal Fisher merge** (Matena & Raffel).
- **AlignMerge-style merge:** (G + λ P_A F_A P_A)⁻¹ G Δ, with a Euclidean rank-r projector P_A on the alignment Fisher. The literature review found it substantially overlaps our repair objective, so a judge will expect it.
- **Standard Fisher merge:** θ₀ weighted by general-data factors, which tests whether our safety weight matters.

**A bug fixed in our patch.** SafeTune's `examples/demos/compare_recover.py:66` swapped base and aligned for C-ΔΘ, which reversed the safety direction. Check any C-ΔΘ call copies `positive=aligned, negative=base`.

### Stage 7 — Locate cross-check (CircuitKIT)

**What it gives.**
- **Tiny Aya is explicitly supported:** `SUPPORTED_MODEL_NAMES` includes `CohereLabs/tiny-aya-global` (`src/circuitkit/backends/_tl_compat/cohere.py:104-110`). The port handles GQA, the parallel block, the bias-free LayerNorm, RoPE only on sliding layers, and the 4096 sliding window.
- **Validation gap:** TL-vs-HF parity was validated on **tiny-aya-base**, not global, with no padded-batch test. Run `CIRCUITKIT_PARITY_MODEL=CohereLabs/tiny-aya-global pytest tests/backends/tl_compat/test_cohere_parity.py` first (the environment variable comes from our patch).
- **Node-level EAP-IG** gives a signed score for each of 16 query heads × 36 layers (576) and each of 36 MLPs. Entry point: `ck.discover(model, task, algorithm="eap-ig", level="node", ...)` (`src/circuitkit/quick.py:465`), or call `get_scores_eap_ig` directly (`src/circuitkit/backends/eap/attribute_node.py:1234`).
- **Score extraction:** `extract_node_head_scores` and `extract_node_mlp_scores` (`applications/quantization/score_extractor.py`); `aggregate_to_kv_heads` (`applications/pruning/score_extractor.py:232`) to compare with our per-KV-group V/O leverage.
- **`safety_units` / `layer_suggestions` exist,** but they are the top 20% of nodes repackaged for SafeTune targeting (`artifacts/scores.py:42-109`), not a ranking. Saved JSON scores are absolute values; signs live only on the live graph.
- **`ck.faithfulness(model, circuit, task, ...)`** runs patching, ablation and random-node baselines (`quick.py:567`). There is no random-*layer* control; build that ourselves.

**What we write.**
- **A custom refusal task.** The built-in safety tasks (advbench, harmbench, …) put the same prompt on both sides, which EAP rejects. Register our own TaskSpec with `ck.register_task` (`tasks/registry.py:14`). It needs `name`, `validate_discovery_config`, `build_dataloader` and `metric_fn()`, and batches are `(clean_list, corrupt_list, LongTensor[B, 2])`.
- **Length-matched pairs.** Each clean (harmful) and corrupt (harmless) prompt must have exactly the same token length after templating. Unequal pairs are end-padded with only a warning. With equal lengths and batch size 1, the shared 366-token preamble has identical activations on both sides and contributes nothing.
- **The metric:** the log-prob of a refusal first token minus a compliance first token at the last position. Check that the refusal tokens for Hindi and Hinglish are single tokens in the 262k vocabulary.

**Gotchas.**
- **Memory.** `discover()` always turns on `use_split_qkv_input` (`api.py:1328-1357`). That copies the residual stream three times per layer and probably runs out of memory at 512 tokens on an L4. Call `get_scores_eap_ig` directly with `model.cfg.use_split_qkv_input = False`, at batch size 1. The CircuitKIT reviewer estimated 2–5 s per pair; that is unmeasured.
- **MLP-score bug, fixed in our patch.** For MLP nodes CircuitKIT read the gradient at the MLP's *input* hook (`attribute_node.py:170`), while heads used the output hook. Our patch moves it to `node.out_hook`, as in upstream EAP-IG. On a toy model the old version agreed in sign with the true patching effect only about half the time. **With the patched install, MLP and head scores are both usable; any MLP scores computed before it are wrong.**
- Scores accumulate in bf16, so small-score ranks are noisy. Report Spearman with bootstrap intervals (scipy; CircuitKIT has no helper).
- The CSV route (`Pipeline.from_custom_data`) silently drops length-mismatched pairs, double-adds BOS to templated text, and caches by name, count and seed. Avoid it.
- **Closest example:** `examples/case-studies/23-jailbreak-refusal-multi-model.ipynb` (harmful vs benign minimal pairs, node-level EAP-IG, batch 1, on 1–2B models).

**Circuit-restricted baselines,** optional:
- `CircuitWeightSteering` (`applications/steering/weight_steering.py`) is heads only and keeps two full models on the GPU.
- `select_components` / `random_selection` + `run_finetuning` (`applications/selective_finetuning/`) work on the HF model, with a built-in random control.

### Stage 8 — Evaluation (AuditKIT)

**What it gives.**
- **Generation:** `ak.generate(samples, model=..., config=ak.RunConfig(max_tokens=512, temperature=0.0, seed=0))` (`src/auditkit/api.py:330`).
  - Load the model with `AutoModel.resolve("hf:<id or local/PEFT dir>", dtype=torch.bfloat16, batch_size=16)`. Without `batch_size` there is no batching. Passing `name=` raised a duplicate-keyword error before our patch.
  - The chat template is always applied, and a plain prompt becomes one user turn.
- **Harmfulness:** `ak.GuardJudge("llama_guard", judge_model=...)` (`src/auditkit/metrics/guard.py:305`).
  - Llama Guard 3 is supported out of the box. It receives the prompt as the user turn and the completion as the assistant turn.
  - It returns 1.0 = unsafe, plus categories. Unreadable replies are excluded from means.
- **Refusal and over-refusal:** `ak.LLMJudge(judge_model="hf:CohereLabs/aya-expanse-8b", prompt=RUBRIC, choices={...})` (`src/auditkit/metrics/judge.py:201`).
  - The prompt can use `{input}`, `{output}` and any metadata key such as `{lang}`, and the verdict is read from the last `CHOICE:` line.
  - Its `system_prompt` is pasted into the user message, so Aya Expanse never sees a system role.
- **Labels:** score precomputed generations with `ak.evaluate(gens, model="precomputed", scorers=[...])`. Per-sample labels are in `prediction.metadata["scores"]`.
- **Global-MMLU:** `MCQAdapter("mcq_loglikelihood")` with `choices=[" A"," B"," C"," D"]`. It skips the chat template and does one forward pass per choice.

**What we write.**
- **Statistics.** AuditKIT has means and normal-approximation standard errors, a paired bootstrap that gives only a p-value (`_bootstrap.py`), and binary Cohen's κ (`ak.judge_calibration`). Wilson intervals, exact McNemar, Holm and κ confidence intervals are absent, so they live in our `stats.py`. `RunComparison.sample_summary` gives flip counts, but "correct" means value == 1.0, which for the guard means *unsafe*.
- **Batched judging.** Judges call the model once per sample, one after another. For speed, batch `HFGenModel(..., batch_size=N).generate(list)` ourselves and reuse `GUARD_PROFILES["llama_guard"]["parse"]`.
- **A refusal rubric per language,** validated against 150 human labels each. There is no refusal metric, and `RefusalDetector` is an English regex.

**Gotchas (these matter).**
- **Memory: one model at a time.** Inside one `evaluate()`, the target stays loaded while the judge loads lazily, about 23 GB on a 24 GB card. Generate, then `.unload()` the target, then run the guard, then `.unload()`, then run the Aya judge.
- **Raw completions: set `AUDITKIT_DISABLE_RUN_CACHE=1`.** Unpatched, every run writes full JSON, raw outputs included, to `$XDG_CACHE_HOME/auditkit/runs/` with no way to stop it. Our patch adds this switch, which skips the cache's reads and writes. Never pass `experiment_name=` and never call `RunResult.save`, which still write raw outputs.
- **Never use `RunResult.push_to_hub` or `CompareResult.push_to_hub`;** they upload `raw_output`. Publish our own cleaned rows (ids, labels, sha256 of the completion) with `auditkit.hf_publish.push_rows_to_hub` or `huggingface_hub`. Strip judge `reason`/`raw` fields too, since they can quote the completion.
- **Stale cache: fixed for local checkpoints by our patch.** The run fingerprint used only the model *path*; it now includes the weight files' sizes and modification times. A judge given as a string spec still goes by its path, so keep judges fixed.
- **There are no multi-turn jailbreak suites in 1.1.6,** so skip the five-turn comparison with Lexsi's Gemma repair.
- SafeTune's own `evaluate()` is English-only (English judges, refusal phrase matching). Use it at most as an English cross-check.

### Stage 9 — Publishing

- Upload whole folders with `huggingface_hub.upload_folder`. CuratorKIT's `push_to_hub` uploads one file and drops the manifest and provenance (`curatorkit/curator.py:588-631`).
- Write dataset and model cards ourselves: licence (Tiny Aya CC-BY-NC 4.0 + LSAL notice), languages, content warning, splits, provenance.
- Make collections with `huggingface_hub`; no repo supports them.
- Check every repo is public and un-gated from a logged-out browser.

---

## 3. What this changes in the plan

1. **Environment pins:** transformers `5.21.1` in the main environment, `circuitkit` in its own, packages installed from the commits above.
2. **Drift fine-tunes:** our own PEFT loop (Stage 2), not `PlainSFTTrainer`.
3. **Factor memory:** chunked hooks and per-layer files on disk (Stage 3). Prompt 0 records `intermediate_size` so the exact numbers can be worked out.
4. **Baselines (Prompt 7):** add Safe Delta fed with our A factors, and the AlignMerge-style merge; fix `pre_model` and the C-ΔΘ argument order; tune α per method.
5. **Localisation (Prompt 8):**
   - use the custom refusal task with length-matched pairs at batch size 1, split-qkv off;
   - head and MLP scores are both usable with the patched install (`scripts/setup_lexsi.sh circuit`);
   - add a random-layer control.
6. **Evaluation (Prompt 2):** one model on the GPU at a time, a temporary `XDG_CACHE_HOME`, our own statistics and sanitised exports.
7. **SafeTune PR:** check licence §6 before opening one. A plugin in our own repo is enough.
8. **Same chat template everywhere,** because its preamble carries part of the refusal behaviour.

## 4. Starting the repo

**Patched, not forked (updated 9 Nov).** We fixed the bugs listed in `patches/README.md` in the four libraries. The fixes ship as patch files in our repo, so nobody has to maintain a GitHub fork:
- `scripts/setup_lexsi.sh` clones each library at its pinned commit into `third_party/`, applies `patches/<Repo>.patch`, and installs it in editable mode with `--no-deps`. Every other version still comes from the lock files.
- In the team folder, the same fixes are committed on a `safety-pencil-patches` branch in each clone under `repos/`.
- The patched copies are modified Lexsi software under LSAL. §2 requires marking them as modified, which `patches/README.md` does.
- We publish only the diffs, not the libraries, and we don't send pull requests to Lexsi (SafeTune §6).

**Our own repo** (call it `safety-pencil`) is the one the submission form links:
1. It already exists as `repos/safety-pencil/` in the team folder. It holds this file, `CLAUDE.md`, `PREREGISTRATION.md`, the four `requirements-*` files, `tests/test_math.py`, `verification/`, `.gitignore` and `.env.example`.
2. Make the first commit and push it to a new private GitHub repo with the commands in its `README.md`. Add both teammates. The repo must be public by submission, with a commit history from the hackathon period.
3. Accept the licences on Hugging Face for `CohereLabs/tiny-aya-global`, `CohereLabs/tiny-aya-base`, `meta-llama/Llama-Guard-3-8B` and `CohereLabs/aya-expanse-8b`. Put `HF_TOKEN=hf_...` in `.env` (gitignored).
4. Create the two environments on the GPU machine:
   ```bash
   python3.12 -m venv .venv && . .venv/bin/activate && pip install -U pip uv
   uv pip install -r requirements-main.lock.txt      # resolves to torch 2.15.0, transformers 5.21.1
   bash scripts/setup_lexsi.sh main                  # patched SafeTune, AuditKIT, CuratorKIT
   deactivate
   python3.12 -m venv .venv-circuit && . .venv-circuit/bin/activate && pip install -U pip uv
   uv pip install -r requirements-circuit.lock.txt   # transformer-lens 3.8.0, transformers 5.22.2
   bash scripts/setup_lexsi.sh circuit               # patched CircuitKIT
   ```
   - Both lock files were resolved on 9 Nov for Linux and Python 3.12.
   - If `torch.cuda.is_available()` is False, reinstall torch from the PyTorch index that matches the driver.
   - On Colab, run `pip uninstall -y torchao` first.
5. Open Claude Code in the repo root and paste Prompt 0, then run the smoke tests below.

## 5. First-day smoke tests, in order

1. Main environment: import safetune, auditkit, curatorkit and peft with transformers 5.21.1. (`safetune.recover` and `safetune.interpret` are the public entry points for `apply_*` and `safety_circuit_info`.) Load tiny-aya-global, print the config (`intermediate_size`, `attention_bias`, `layer_types`), and print the chat template for one prompt.
2. Factor hooks on 2 layers with 8 prompts: the shared A has the null vector 1/w, and `gen_eig` succeeds on it.
3. `PencilTrainer` on a 2-layer slice: `step_ratio` ≤ 1, and the result matches `pencil_repair_np`.
4. One SafeTune baseline (`wiseft`, the cheapest) on the same checkpoint, saved to its own directory.
5. AuditKIT: generate 16 prompts, unload, run Llama Guard 3, unload; confirm labels come back and the cache folder is wiped.
6. Circuit environment: the parity test with `MODEL_NAME` set to tiny-aya-global, then EAP-IG on 4 length-matched English pairs with split-qkv off; record memory use and time per pair.
