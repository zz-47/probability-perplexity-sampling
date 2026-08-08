# probability-perplexity-sampling

The full math of token probability, measured on real SLM logits — softmax and cross-entropy to perplexity to temperature/top-k/top-p sampling to scale-out (135M→1.7B) to deploy-time decoding configs. CPU-only, no assumptions.

---

## Studies

| # | Topic | Core formula | Status |
|---|-------|-------------|--------|
| 1 | Softmax, cross-entropy, the numerical floor | `p = softmax(z)`, `CE = logsumexp(z) − z_y` | ✅ Complete (5 experiments measured) |
| 2 | Perplexity as an instrument | `PPL = exp(mean NLL)`, `BPB = Λ/bytes` | 🚧 Scaffolded — protocol sweeps pending |
| 3 | Truncation under synthetic control | closed-form coverage vs measured cut | ⬜ Planned |
| 4 | Truncation operators on real logits | `KL(p′‖p)`, effective support, distinct-n | ⬜ Planned |
| 5 | Scale-out: distribution shape 135M→1.7B | entropy / `top1` / `k90` vs model size | ⬜ Planned |
| 6 | Deployment decoding configs | operator ordering, determinism, ms/token | ⬜ Planned |

---

## Repository Structure

```
probability-perplexity-sampling/
├── README.md                          ← this file
├── requirements.txt                   ← pinned env (torch 2.13.0+cpu, transformers 5.14.1)
├── .venv/                             ← local environment (hidden, gitignored)
├── pps_cache/                         ← cached logit tensors, models never re-enter the kernel
├── 1_softmax_cross_entropy.ipynb      ← the logit→probability map, measured and stress-tested
├── 2_perplexity_instrument.ipynb      ← context length, stride, tokenizer: what moves the scalar
├── 3_truncation_controlled.ipynb      ← (planned) known-tail distributions, closed-form coverage
├── 4_truncation_real_logits.ipynb     ← (planned) top-k / top-p / min-p on real next-token rows
├── 5_scaleout_distribution_shape.ipynb← (planned) does distribution shape travel 135M→1.7B?
└── 6_decoding_deployment.ipynb        ← (planned) ordering, seeds, latency per operator
```

Each notebook is **self-contained**, runs on CPU-only Windows, and follows the *hypothesis → measurement → verdict* narrative. Model weights are never copied into the repository: `HF_HUB_OFFLINE=1` resolves every load against the shared local hub cache.

---

## Study 1 — From logits to probabilities: softmax, cross-entropy, and the numerical floor (complete)

**Design.** One float32 forward pass of SmolLM2-135M over a fixed passage yields `Z ∈ R^{83×49152}`, cached to `pps_cache/logits_135M.pt`. Five experiments read that one tensor: shift invariance and the overflow wall, the three-route cross-entropy identity, the temperature family with its entropy derivative, and the shape of real next-token distributions.

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| H1 | shift invariance holds to float noise | ≤ 1e-7 | **1.192e-07** = float32 eps, worst case over `c ∈ [−100, 100]` | ✅ Holds |
| H2 | naive softmax overflows, stable form does not | `nan` past shifted max 88.72 | `nan` at `c = 60` → shifted max **89.13**; analytic crossing 59.585 | ✅ Holds |
| H3 | three routes to cross-entropy agree | ≤ 1e-6 | gather ≡ `F.cross_entropy` at **0.0**; closed form off **2.97e-05** | ⚠️ Partial |
| H4 | entropy strictly increasing in `T`, ceiling `log V` | monotone, within 0.05 | monotone **True**, gap **0.0005**, `dH/dT = Var/T³` verified to 1e-3 | ✅ Holds |
| H5 | real distributions are far from uniform | median `k90` < 50, `exp(H)` < 200 | `exp(H)` **66.6** ✓, `k90` **222** ✗ | ⚠️ Partial |

**The tail sets the width, not the head.** Median `top1` is 0.259 and median `exp(H)` is 66.6 — from which `k90 < 50` looked safe. Measured `k90 = 222`, a **3.3× gap** between the mass-weighted width and the token count. `exp(H)` is dominated by the head; `k90` must walk the tail. Any operator sized from head statistics will mis-cut.

**Fixed-`k` is impossible, measured on one document.** Position 0 (no context) needs **12,097** tokens to reach 90% mass; position 77 (`"…rather"` → `" than"`) needs **1**. Same passage, same model. This is the structural case for mass-based truncation, and it is a measurement rather than an argument.

**The closed form is not the safe form.** `logsumexp(z) − z_y` is the identity everyone writes on the board and it is ~250× less accurate than gathering from `log_softmax`, because it subtracts two ~34-magnitude numbers to produce a ~4-magnitude answer. `F.cross_entropy` agrees with the gather route bit-for-bit (0.0) — it is the same kernel. Algebraic identity does not imply numerical interchangeability.

**Numerical stability is worth two orders of magnitude, not just safety.** The naive softmax carries a floor of **1.681e-05** even where it stays finite — 141× eps, from summing 49,152 terms spanning 24 orders of magnitude. It sits between the pairwise-summation bound (1.9e-6) and the random-walk estimate (2.6e-5). The overflow wall itself is *predictable* from the data: max logit 33.881 leaves 54.84 of headroom against the float32 `exp` limit of 88.72, and the model's own raw scale already consumes 38% of the exponent budget.

**Temperature has a measurable peak, and it is not at `T = 1`.** The identity `dH/dT = Var_{p_T}(z)/T³` was verified against finite differences to 1e-3. Its right-hand side runs 0.127 → 1.157 → **6.914** → 0.085 across `T ∈ {0.5, 1, 2, 5}`: entropy motion peaks near `T = 2` and dies at both ends, because a collapsed distribution sees no logit disagreement and a flat one is crushed by `T³`. Concretely, `exp(H)` moves 1.3 → 2302.8 between `T = 1` and `T = 2` — a **1,700× change in effective support over one unit of T**. Below `T = 1` the knob does nothing on this row (median spread 34.7 nats leaves nothing to sharpen); above 1.5 it is a cliff.

**Loss measures knowledge, not fluency.** The hardest position (`"The transformer architecture replaced"` → `" recurrence"`, NLL 17.84, `p_y = 1.79e-08`) drew a grammatically flawless wrong answer, `" the"` at 0.484. The easiest (`" than"` after `"rather"`, `p = 0.995`) is a bigram constraint. Four of 82 positions scored worse than a uniform guess (`log V = 10.803`), and because perplexity is the geometric mean of `1/p_y`, those four alone pull the passage figure of **76.94** far above what the typical token (median NLL 3.61) deserves.

**Verdict in one line.** The training loss and the sampling distribution are the same 49,152-way object read two ways, and its practically relevant width is set by its tail: median `exp(H)` 66.6 against median `k90` 222, ranging from 1 to 12,097 inside a single document.

---

## Study 2 — Perplexity as an instrument: what the number depends on besides the model (scaffolded)

**Question.** Perplexity is reported as a property of a model. It is a property of a model *and* a tokenizer *and* a context length *and* a stride *and* a corpus. Hold the model fixed — literally the same cached weights — move everything else, and measure how far the reported scalar travels.

**Design.** Four experiments on one scorer. A `score(ids, L, S)` rig walks the sequence in windows of `L` at stride `S`, scoring each token **exactly once** (overlapping context, never overlapping scored positions — the double-counting bug is asserted against in the rig itself). Then: a context-length sweep at stride 1, a stride sweep at fixed window, a cross-tokenizer comparison against DistilGPT2 (50,257-way GPT-2 BPE) on the identical raw string, and a calibration test of perplexity against the model's own `exp(mean H)`.

**Pre-committed hypothesis board:** PPL falls monotonically with context (H1) and saturates before the full window (H2); disjoint chunking overstates PPL by ≥ 1.3× against stride-1 (H3); per-token PPL differs across tokenizers by ≥ 20% (H4) and bits-per-byte closes most of that gap (H5); the model is over-confident on this text, `PPL > exp(mean H)` (H6).

**The math the study rests on.** `PPL = exp(−(1/N)Σ log q(x_t|x_{<t}))` is the geometric mean of `1/q(x_t)`, so it is dominated by the worst positions rather than the typical one. Conditioning on less cannot help on average (`E[−log q(x_t|x_{t−L+1:t−1})] ≥ E[−log q(x_t|x_{<t})]`), which forces H1's direction but says nothing about the saturation point. With window `L` and stride `S`, scored tokens average `L − (S+1)/2` tokens of context at a cost of `N/S` forward passes — the entire stride trade in one line. And since the token count `N` is a tokenizer's choice, only `BPB = Λ/bytes = (N/B)·log₂(PPL)` is comparable across segmentations.

**Stated confound.** DistilGPT2 differs in weights and training data as well as tokenizer, so the residual BPB gap is not attributable to segmentation. The narrower claim under test — that the **PPL gap overstates the BPB gap** — is valid despite the confound, and is the reason BPB is the quantity being defended.

**Status:** scaffold written, code cells open, hypothesis board pre-committed.

---

## Caveats carried across the repository

- **One passage, one primary model.** Absolute numbers (`k90`, `exp(H)`, perplexity) describe this text under SmolLM2-135M. Ratios and identities are the transferable content.
- **Positions are not i.i.d.** They come from a single document, so medians are descriptive, not corpus statistics.
- **float32 throughout.** Deliberate: several claims are about float32 boundaries. bf16 inference would move the numerical findings and not the structural ones.
- **Final-layer logits only.** No intermediate-layer probing; the object under study is the head's output.
- **Model weights live in the shared HF hub cache.** Nothing is copied into the repository; `HF_HUB_OFFLINE=1` guarantees no network access at run time.
