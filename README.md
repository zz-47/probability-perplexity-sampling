# probability-perplexity-sampling

The full math of token probability, measured on real SLM logits — softmax and cross-entropy to perplexity to temperature/top-k/top-p sampling to scale-out (135M→1.7B) to deploy-time decoding configs. CPU-only, no assumptions.

---

## Studies

| # | Topic | Core formula | Status |
|---|-------|-------------|--------|
| 1 | Softmax, cross-entropy, the numerical floor | `p = softmax(z)`, `CE = logsumexp(z) − z_y` | ✅ Complete (5 experiments measured) |
| 2 | Perplexity as an instrument | `PPL = exp(mean NLL)`, `BPB = (N/B)·log₂PPL` | ✅ Complete (5 experiments measured) |
| 3 | Truncation under synthetic control | `D_KL(p′‖p) = −log(mass_kept)` | 🚧 Scaffolded — oracle sweeps pending |
| 4 | Truncation operators on real logits | `KL(p′‖p)`, effective support, distinct-n | ⬜ Planned |
| 5 | Scale-out: distribution shape 135M→1.7B | entropy / `top1` / `k90` vs model size | ⬜ Planned |
| 6 | Deployment decoding configs | operator ordering, determinism, ms/token | ⬜ Planned |

---

## Repository Structure

```
probability-perplexity-sampling/
├── README.md                          ← this file
├── NOTATION.md                        ← every symbol, metric, and named operation, defined once
├── requirements.txt                   ← pinned env (torch 2.13.0+cpu, transformers 5.14.1)
├── .venv/                             ← local environment (hidden, gitignored)
├── pps_cache/                         ← cached logit tensors, models never re-enter the kernel
├── 1_softmax_cross_entropy.ipynb      ← the logit→probability map, measured and stress-tested
├── 2_perplexity_instrument.ipynb      ← context length, stride, tokenizer: what moves the scalar
├── 3_truncation_controlled.ipynb      ← known-tail distributions, closed-form coverage vs measured cut
├── 4_truncation_real_logits.ipynb     ← (planned) top-k / top-p / min-p on real next-token rows
├── 5_scaleout_distribution_shape.ipynb← (planned) does distribution shape travel 135M→1.7B?
└── 6_decoding_deployment.ipynb        ← (planned) ordering, seeds, latency per operator
```

Each notebook is **self-contained**, runs on CPU-only Windows, and follows the *hypothesis → measurement → verdict* narrative. Model weights are never copied into the repository: `HF_HUB_OFFLINE=1` resolves every load against the shared local hub cache.

**Claim labels are `C1 … C6`**, not `H1 … H6`, so that `H` unambiguously means Shannon entropy — which appears as a measured quantity in several studies. Full symbol table in `NOTATION.md`.

---

## Study 1 — From logits to probabilities: softmax, cross-entropy, and the numerical floor (complete)

**Design.** One float32 forward pass of SmolLM2-135M over a fixed passage yields `Z ∈ R^{83×49152}`, cached to `pps_cache/logits_135M.pt`. Five experiments read that one tensor: shift invariance and the overflow wall, the three-route cross-entropy identity, the temperature family with its entropy derivative, and the shape of real next-token distributions.

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | shift invariance holds to float noise | ≤ 1e-7 | **1.192e-07** = float32 eps, worst case over `c ∈ [−100, 100]` | ✅ Holds |
| C2 | naive softmax overflows, stable form does not | `nan` past shifted max 88.72 | `nan` at `c = 60` → shifted max **89.13**; analytic crossing 59.585 | ✅ Holds |
| C3 | three routes to cross-entropy agree | ≤ 1e-6 | gather ≡ `F.cross_entropy` at **0.0**; closed form off **2.97e-05** | ⚠️ Partial |
| C4 | entropy strictly increasing in `T`, ceiling `log V` | monotone, within 0.05 | monotone **True**, gap **0.0005**, `dH/dT = Var/T³` verified to 1e-3 | ✅ Holds |
| C5 | real distributions are far from uniform | median `k90` < 50, `exp(H)` < 200 | `exp(H)` **66.6** ✓, `k90` **222** ✗ | ⚠️ Partial |

**The tail sets the width, not the head.** Median `top1` is 0.259 and median `exp(H)` is 66.6 — from which `k90 < 50` looked safe. Measured `k90 = 222`, a **3.3× gap** between the mass-weighted width and the token count. `exp(H)` is dominated by the head; `k90` must walk the tail. Any operator sized from head statistics will mis-cut.

**Fixed-`k` is impossible, measured on one document.** Position 0 (no context) needs **12,097** tokens to reach 90% mass; position 77 (`"…rather"` → `" than"`) needs **1**. Same passage, same model. This is the structural case for mass-based truncation, and it is a measurement rather than an argument.

**The closed form is not the safe form.** `logsumexp(z) − z_y` is the identity everyone writes on the board and it is ~250× less accurate than gathering from `log_softmax`, because it subtracts two ~34-magnitude numbers to produce a ~4-magnitude answer. `F.cross_entropy` agrees with the gather route bit-for-bit (0.0) — it is the same kernel. Algebraic identity does not imply numerical interchangeability.

**Numerical stability is worth two orders of magnitude, not just safety.** The naive softmax carries a floor of **1.681e-05** even where it stays finite — 141× eps, from summing 49,152 terms spanning 24 orders of magnitude. It sits between the pairwise-summation bound (1.9e-6) and the random-walk estimate (2.6e-5). The overflow wall itself is *predictable* from the data: max logit 33.881 leaves 54.84 of headroom against the float32 `exp` limit of 88.72, and the model's own raw scale already consumes 38% of the exponent budget.

**Temperature has a measurable peak, and it is not at `T = 1`.** The identity `dH/dT = Var_{p_T}(z)/T³` was verified against finite differences to 1e-3. Its right-hand side runs 0.127 → 1.157 → **6.914** → 0.085 across `T ∈ {0.5, 1, 2, 5}`: entropy motion peaks near `T = 2` and dies at both ends, because a collapsed distribution sees no logit disagreement and a flat one is crushed by `T³`. Concretely, `exp(H)` moves 1.3 → 2302.8 between `T = 1` and `T = 2` — a **1,700× change in effective support over one unit of T**. Below `T = 1` the knob does nothing on this row (median spread 34.7 nats leaves nothing to sharpen); above 1.5 it is a cliff.

**Loss measures knowledge, not fluency.** The hardest position (`"The transformer architecture replaced"` → `" recurrence"`, NLL 17.84, `p_y = 1.79e-08`) drew a grammatically flawless wrong answer, `" the"` at 0.484. The easiest (`" than"` after `"rather"`, `p = 0.995`) is a bigram constraint. Four of 82 positions scored worse than a uniform guess (`log V = 10.803`), and because perplexity is the geometric mean of `1/p_y`, those four alone pull the passage figure of **76.94** far above what the typical token (median NLL 3.61) deserves.

**Verdict in one line.** The training loss and the sampling distribution are the same 49,152-way object read two ways, and its practically relevant width is set by its tail: median `exp(H)` 66.6 against median `k90` 222, ranging from 1 to 12,097 inside a single document.

---

## Study 2 — Perplexity as an instrument: what the number depends on besides the model (complete)

**Design.** The model is frozen — literally the same cached weights — and only the instrument moves. A `score(ids, L, S)` rig walks the sequence in end-anchored windows of `L` at stride `S`, scoring each token **exactly once** and asserting `n_scored == N − 1` and `maxwin ≤ L` on every call. Four experiments follow: a context-length sweep at stride 1, a stride sweep at `L = 128`, a cross-tokenizer comparison against DistilGPT2 on the identical raw string, and a calibration test of perplexity against the model's own `exp(mean H)`. Passage: 486 SmolLM2 tokens / 2,628 bytes.

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | PPL falls monotonically with context length | monotone decreasing | strictly monotone `L = 8 → 256`; **−0.3% at full context** | ⚠️ Partial |
| C2 | The curve saturates before the full window | most gain by `L ≈ 64` | still improving **9% per doubling at `L = 256`** | ❌ Reversed |
| C3 | Disjoint chunking overstates PPL | ≥ 1.3× stride-1 | **1.214×** (38.096 vs 31.377) | ⚠️ Partial |
| C4 | Per-token PPL is not cross-tokenizer comparable | ≥ 20% apart | **2.562×** apart | ✅ Holds (wrong mechanism) |
| C5 | BPB closes the cross-tokenizer gap | BPB nearer 1 than PPL | **1.301× vs 2.562×** | ✅ Holds (wrong mechanism) |
| C6 | The model is over-confident on this text | `PPL > exp(mean H)` | **28.871 vs 28.116**, +0.0265 nats | ✅ Holds, marginally |

**Protocol moves the number more than the model does.** Total swing from `L = 8` to full context is **3.53×** — larger than the **2.562×** gap this same study measures between SmolLM2-135M and DistilGPT2, two entirely different models. A perplexity quoted without its context length, stride, and tokenizer is not a measurement of a model.

**There is no saturation (C2 reversed).** NLL drops 0.609, 0.298, 0.162, 0.109, 0.087 per doubling of `L`: **each doubling buys ~60% of what the previous doubling bought**, a smooth geometric decay still running 9% per doubling at `L = 256`. The prediction that context stops helping by `L ≈ 64` was wrong by a wide margin — `L = 64` sits 21% above the best measured value, and the doublings past it still buy 11.5% then 9%. The model draws usable signal from tokens 64 to 256 positions back; it is not a 64-gram. Because the gains shrink at a constant *rate* rather than hitting a floor, there is no principled place to stop — only a cost curve. Long-range mutual information in ordinary prose is small but never zero.

**The stride knee is sharp, and stride-1 is not the gold standard.** Disjoint chunking costs 1.214×, but the gap closes almost immediately: `S = 16` recovers **93.1% at 6.4% of stride-1's compute**, `S = 8` reaches 99.1% at 12.6%, `S = 4` recovers 100% at 25%. Stride-1 itself lands at **98.1%** — worse than `S = 4` and `S = 8` while costing four to eight times as much.

**Two independent inversions.** `L = 256` beats full context by 0.3%, and `S = 4` beats `S = 1` by 0.4%. The first is explained: the full-context row averages over a 1-to-485 context ramp while the strided rows sit nearly flat at 255, so they are different protocols and the conditioning inequality does not apply between them. The second has no such explanation and stands as measured. One passage cannot separate genuine long-range interference from position-specific noise, so it is reported rather than resolved.

**Bits-per-byte corrected almost nothing here, and that is the finding.** Two independently trained BPE vocabularies produced nearly identical segmentations — 486 tokens against 494, **fertility ratio 1.016×**, differing in one boundary out of the first fourteen (`ĠT`/`rained` vs `ĠTr`/`ained`). Convergence is the expected outcome: BPE is a greedy frequency-merge procedure, so two runs over comparable English corpora discover nearly the same merges in nearly the same order. So the 2.562× PPL gap was never a tokenizer artifact; it was model quality all along. The BPB ratio of 1.301× decomposes as `log₂(73.965)/log₂(28.871) = 1.280` times fertility `1.016` — **the logarithm supplies 98% of the apparent correction**. BPB remains the right unit (a character-level tokenizer would show fertility 4–5× and the correction would be real), but on this pair, saying BPB "removed the tokenizer effect" would be reporting a change of units.

**Calibration is set by the tail, not the centre.** In aggregate the model is over-confident by **+0.0265 nats** (2.7%). At the median position it is *under*-confident by more than a nat — median NLL **2.433** against median H **3.509** — and only **37.3%** of positions are over-confident at all. Both quantities are averages in log space, so a small minority of catastrophic positions erases the accumulated slack of hundreds of well-hedged ones. A model tuned to improve mean calibration would be tuning against a few dozen positions out of five hundred. Plainly: the model knows what it does not know most of the time, and is spectacularly wrong a few times — and averaging in log space lets those few decide the headline.

**A rig correction worth recording.** The scaffold predicted mean context `L − (S+1)/2` = 127 at `L = 128, S = 1`; measured **110.5**. The formula is steady-state only — the first 127 targets sit in a window pinned at `begin = 0` and receive 1, 2, … 127 tokens rather than 127 each. Exact accounting `(127·128/2 + 127·358)/485 = 110.5` reproduces it. Separately, the rig's index algebra was verified by exhaustive simulation over the full `(L, S)` grid before use, which is how the constraint `S ≤ L − 1` surfaced: a window of `L` tokens has no predictor for its own first token, so **disjoint chunking is `S = L − 1`**, not `S = L`.

**Verdict in one line.** Holding the model perfectly fixed, protocol alone moves the reported perplexity by 3.53× — more than the gap between two different models — and the stride cost curve says `S ≈ L/8` buys 93% of the sliding-window benefit for 6.4% of the compute.

---

## Study 3 — Truncation under synthetic control (scaffolded)

**Question.** On real logits, nobody knows what a truncation operator removed: the true distribution is unavailable, so a cut can be described but never scored. Build the distribution first — with a designated signal head and a noise tail of exactly known mass — and the correct cut becomes a number.

**Design.** Zipf-shaped distributions over a 49,152-symbol alphabet, symbol order shuffled under a fixed seed so head membership is recoverable only from the probabilities. Three cases: **peaked** (`n_sig = 5`, `m_sig = 0.95`), **typical** (`n_sig = 250`, `m_sig = 0.85`, `s_head = 1.4` — grid-searched to land at `exp(H) ≈ 61`, `k90 ≈ 240` against the measured real-logit medians of 66.6 and 222), and **flat** (`n_sig = 5000`). Top-k, top-p, and min-p are implemented from their definitions and scored on `mass_kept`, `n_kept`, `noise_removed`, `signal_lost`, `D_KL`, and `exp(H′)`. Float64 throughout, no model loaded.

**The math the study rests on.** For any pure truncation with retained mass `M`, renormalization gives `p′_i = p_i/M` on the support, so

`D_KL(p′‖p) = Σ_{i∈S} (p_i/M)·log((p_i/M)/p_i) = −log M`

The distortion depends on **nothing but the retained mass** — two operators keeping 90% are equally "distorted" even if one kept pure signal and the other pure noise. KL measures how much was cut, never whether the right thing was cut, which is precisely why oracle-based scores are carried alongside it. The reverse direction `D_KL(p‖p′)` is infinite for any truncation.

**Pre-committed board:** the KL identity holds to 1e-12 in float64 (C1); all three operators can be tuned to the oracle cut on a fixed distribution (C2); a fixed `k` fails under a shape change with `noise_removed` spread > 0.4 (C3); top-p holds mass while its count swings > 10× (C4); min-p has the smallest spread (C5); top-p degenerates to `n_kept > 1000` on the flat case (C6).

**Status:** scaffold written with full pseudocode structure per experiment, code cells open, board pre-committed.

---

## Caveats carried across the repository

- **One passage per study, one primary model.** Absolute numbers (`k90`, `exp(H)`, perplexity, BPB) describe the specific text under SmolLM2-135M. Ratios and identities are the transferable content.
- **Positions are not i.i.d.** They come from a single document, so medians are descriptive, not corpus statistics.
- **float32 throughout studies 1–2, float64 in study 3.** Deliberate: several claims are about float32 boundaries, while study 3's exactness claim would be untestable at float32.
- **Final-layer logits only.** No intermediate-layer probing; the object under study is the head's output.
- **Cross-model comparisons confound weights with tokenizer.** DistilGPT2 differs from SmolLM2 in training data and parameters as well as segmentation; only the narrow claims are drawn.
- **Model weights live in the shared HF hub cache.** Nothing is copied into the repository; `HF_HUB_OFFLINE=1` guarantees no network access at run time.
