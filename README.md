# probability-perplexity-sampling

The full math of token probability, measured on real SLM logits — softmax and cross-entropy to perplexity to temperature/top-k/top-p sampling to scale-out (135M→1.7B) to deploy-time decoding configs. CPU-only, no assumptions.

---

## Studies

| # | Topic | Core formula | Status |
|---|-------|-------------|--------|
| 1 | Softmax, cross-entropy, the numerical floor | `p = softmax(z)`, `CE = logsumexp(z) − z_y` | ✅ Complete (5 experiments measured) |
| 2 | Perplexity as an instrument | `PPL = exp(mean NLL)`, `BPB = (N/B)·log₂PPL` | ✅ Complete (5 experiments measured) |
| 3 | Truncation under synthetic control | `D_KL(p′‖p) = −log(mass_kept)` | ✅ Complete (5 experiments measured) |
| 4 | Truncation operators on real logits | `KL(p′‖p)`, `n_kept` spread, `T`∘`Tr` ordering | ✅ Complete (4 experiments measured) |
| 5 | Scale-out: distribution shape 135M→1.7B | entropy / `top1` / `k90` vs model size | ✅ Complete (3 experiments measured) |
| 6 | Deployment decoding configs | operator ordering, determinism, ms/token | 🚧 Scaffolded — 3 experiments pending |

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
├── 4_truncation_real_logits.ipynb     ← same operators on real rows: does the ordering survive?
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

## Study 3 — Truncation under synthetic control: what top-k, top-p and min-p actually remove (complete)

**Design.** Two-block Zipf distributions over a 49,152-symbol alphabet with a designated signal head and a noise tail of exactly known mass, symbol order permuted under a fixed seed so head membership is recoverable only from probability values. Three cases: **peaked** (`n_sig = 5`, `exp(H) = 5.7`), **typical** (`n_sig = 250`, `m_sig = 0.85`, `s_head = 1.4` — grid-searched to `exp(H) = 61.1`, `k90 = 240` against measured real-logit medians of 66.6 and 222), and **flat** (`n_sig = 5000`, `exp(H) = 4907`). The three operators are written from their definitions and scored on `mass_kept`, `n_kept`, `noise_removed`, `signal_lost`, `D_KL` and `exp(H′)`. Float64, no model loaded — the only model-free study in the repository, which is what makes its ground truth exact.

**Measured findings (synthetic, ground truth known):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | `D_KL(p′‖p) = −log(mass_kept)` | ≤ 1e-12 float64 | **3.33e-16** over 10 settings; float32 **1.54e-07** | ✅ Holds |
| C2 | All three tunable to the oracle cut | `noise_rm > 0.95`, `sig_lost < 0.05` | best **0.7219 / 0.6902 / 0.6701**; best `sig_lost` **0.1779** | ❌ Reversed — all fail |
| C3 | Fixed `k` fails under shape change | spread > 0.4 | **0.9608** (0.0392 → 1.0000) | ✅ Holds |
| C4 | Top-p holds mass, not count | `n_kept` > 10×, mass < 0.01 | **355.3×** (10 → 3,553), mass range **0.0072** | ✅ Holds |
| C5 | Min-p adapts best | smallest spread | **top-p 0.4941** < min-p 0.6079 < top-k 0.9608 | ❌ Reversed |
| C6 | Top-p degenerates on flat | `n_kept` > 1,000 | **4,815** of 49,152 | ✅ Holds |

**Three conventional defaults, one distribution, a 48× spread.** On the typical case, `min_p = 0.1` keeps **5** candidates, `top_k = 50` keeps **50**, `top_p = 0.9` keeps **240**. Anyone switching a serving config between these moves the surviving candidate set by nearly two orders of magnitude without changing anything they would describe as a policy. `exp(H)` falls from 61.1 to 3.6 / 13.1 / 24.4 respectively — the mechanical reason sampled text changes character when these knobs move.

**C2 reversed, and the failure is structural rather than a tuning problem.** No setting of any operator recovers the designed head: best scores 0.722, 0.690, 0.670 against a possible 1.0, with the best achievable signal loss at 0.1779 versus a 0.05 bar. The reason is that the head is itself Zipf-shaped, so its weakest members carry *less* probability than the tail's strongest members — the two sets **interleave in probability**. Every one of these operators is a single threshold (on rank, on cumulative mass, or on height relative to the peak), and no threshold separates interleaved sets. They are not failing to find a boundary that exists; there is no boundary to find in the only coordinate they can read. That is the strongest available argument for why decoding heuristics remain heuristics.

**C5 reversed — min-p's adaptivity points the wrong way.** Predicted most stable, measured **second worst**: `noise_removed` spreads of top-p 0.4941, min-p 0.6079, top-k 0.9608. Min-p thresholds at `α·max(p)`, so its cut tracks the peak; on a flat distribution the peak is small and the *relative* rule keeps only 46 symbols while discarding **53.6% of the head**. It becomes more aggressive exactly where the distribution is widest and the model least certain. Across the sweep its `n_kept` collapses 46 → 4 while `signal_lost` never falls below 0.17.

**Top-p's real guarantee, found rather than predicted.** It records `signal_lost = 0.0000` on the three flattest cases. Being mass-based, it cannot cut into the head while the head is diffuse — the head's mass is exhausted before its symbols are. That structural property is why its noise-removal spread is smallest despite its count varying 355×. The price is C6: on the flattest case it keeps 4,815 symbols, so a nucleus is not a nucleus when entropy is high.

**Reading the three as a design space.** Each pins one quantity and lets the others float — top-k pins count and loses control of mass *and* of noise removal; top-p pins mass and keeps the most control of noise; min-p pins relative height and controls neither. Of the three invariants, **mass is the one worth having**, not because it is intrinsically better but because on Zipf-like distributions it is the only one whose failure mode is bounded.

**KL cannot arbitrate, and the identity proves it.** `D_KL(p′‖p) = −log M` holds to 3.33e-16, so KL is a monotone function of retained mass and nothing else. In the defaults table min-p has the **highest** KL (0.592) and the **best** noise removal (1.0000); top-p has the **lowest** KL (0.105) and the **worst** (0.586). Ranking operators by divergence ranks them by how much they cut — the parameter, not the outcome. The reverse direction is worse than uninformative: 48,912 of 49,152 coordinates have `p > 0` but `p′ = 0`, so `D_KL(p‖p′)` is undefined rather than merely large.

**A precision result carried from the first study.** The KL identity holds to 3.33e-16 in float64 but only **1.54e-07** in float32 — nine orders of magnitude, just above eps, from summing 49,152 terms spanning many orders of magnitude. Same mechanism as the naive-softmax floor of 1.681e-05. Vocabulary size sets the precision floor, not the operator, and real decoding stacks run this arithmetic in float32 or bf16.

**Verdict in one line.** No operator recovers the head at any setting because head and tail interleave in probability; among the three, the one marketed as adaptive is measurably the second least stable under a change of shape, and the mass-based rule is the only one with a bounded failure mode.

---

## Study 4 — Truncation on real logits: does the controlled ordering survive? (complete)

**Measured findings (real SmolLM2-135M logits, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | Fixed `k = 50` varies across positions | `mass_kept` spans > 0.5 | **0.8244** (0.174 → 0.999) | ✅ Holds |
| C2 | Top-p's count varies more than synthetic | `n_kept` spread > 355× | **12,097×** (1 → 12,097) | ✅ Holds |
| C3 | Controlled ordering transfers | top-p < min-p < top-k | top-p 0.095 < **top-k 0.824** < min-p 0.863 | ❌ Reversed |
| C4 | Min-p most aggressive at high entropy | negative correlation | Spearman **+0.457** | ❌ Reversed |
| C5 | Ordering matters less than operator choice | ordering TVD < between-op TVD | ordering 0.416 > min between-op 0.170 | ❌ Reversed |
| C6 | All keep < 25% of V at widest position | each keeps < 25% | top-k 0.10%, top-p 24.6%, min-p 0.05% | ✅ Holds |

**Top-p is the winner on stability and it's not close.** Mass spread **0.095** vs top-k 0.824 / min-p 0.863 — roughly 9× tighter. **C1 ✅** (top-k mass spread 0.824). **C2 ✅** (top-p `n_kept` swings 12,097× vs synthetic 355×). **C6 ✅** (all under 25% at the widest position).

**C3/C4/C5 all reverse, two of them sharply.** The controlled study's ordering (top-p < min-p < top-k) does *not* transfer: top-k (0.824) beats min-p (0.863), so the measured ordering is top-p < top-k < min-p. Min-p's negative correlation was specific to synthetic flat distributions; on real data it is **+0.457** (weakly adaptive, not inverted). And operator ordering matters *more* than operator choice at high temperature: at T=1.5 the temperature-then-cut vs cut-then-temperature supports differ at all 83 positions (mean TVD 0.416), exceeding the smallest between-operator distance (0.170).

**Top-p's real guarantee, and its real cost.** Spearman(k90, n_kept) = **+1.000** — it widens exactly where the distribution widens, the only operator whose count actually tracks what each position needs. But that adaptivity is the 12,097× swing: at the widest position top-p hands the sampler **12,097 candidates (24.6% of the vocabulary)**.

**Top-k's rank invariance, confirmed.** Temperature is rank-preserving, so top-k picks the same 50 tokens whether you apply T before or after the cut — **0/83** supports differ at any T. That is a structural guarantee the other two operators do not have.

**Verdict in one line.** On real logits the mass-based rule is still the most stable across positions, but three of six predictions reverse — most sharply the claim that operator ordering matters less than operator choice.

---

## Study 5 — Scale-out: does distribution shape travel 135M → 1.7B? (complete)

**Measured findings (SmolLM2 135M, 360M, 1.7B, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | Width grows with size | median `k90`, `exp(H)` rise | k90 222→38→22, exp(H) 66.6→17.4→11.2 | ❌ Reversed |
| C2 | Concentration falls | median `top1` declines | top1 0.259→0.360→0.422 | ❌ Reversed |
| C3 | PPL falls | lower at 1.7B | 76.94→17.52→12.53 | ✅ Holds |
| C4 | Shape is size-dependent | `exp(H)`/`k90` ratio changes | 0.371→0.534→0.659 | ✅ Holds |

**Bigger models are MORE concentrated, not less.** Every width statistic moves the wrong way against the predictions. The 135M model spreads its probability over ~222 tokens to reach 90% mass; the 1.7B model needs only 22. The leading token's share grows from 0.259 to 0.422. Entropy falls from 4.20 to 2.42 nats. The vocabulary is identical — the bigger model simply uses less of it.

**PPL falls as expected, but alongside a fundamentally different shape.** PPL drops 76.94 → 17.52 → 12.53 (0.163× the 1.35M value at 1.7B). **C3 ✅.** But this improvement comes *alongside* a distribution that is substantially more concentrated, not the same-shaped distribution at higher resolution.

**The deployment consequence.** Study 4 found that a fixed `k = 50` keeps 17% of the mass at one position and 99.9% at another on 135M logits. At 1.7B the typical `k90` is 22, so `k = 50` already covers nearly all the mass everywhere — the operator becomes nearly a no-op. Conversely, `top_p = 0.9` keeps 22 tokens at the typical 1.7B position (versus 222 at 135M), so the same nucleus is cutting much more aggressively relative to the distribution it meets. **Every default measured in studies 1–4 is model-specific.**

**Verdict in one line.** Bigger models are more concentrated, not less — three pre-registered predictions reverse — while PPL falls as expected, meaning decoding improvement and distributional sharpness travel together but point in opposite directions.

---

## Caveats carried across the repository

- **One passage per study, one primary model.** Absolute numbers (`k90`, `exp(H)`, perplexity, BPB) describe the specific text under SmolLM2-135M. Ratios and identities are the transferable content.
- **Positions are not i.i.d.** They come from a single document, so medians are descriptive, not corpus statistics.
- **float32 throughout studies 1–2, float64 in study 3.** Deliberate: several claims are about float32 boundaries, while study 3's exactness claim would be untestable at float32.
- **Final-layer logits only.** No intermediate-layer probing; the object under study is the head's output.
- **Cross-model comparisons confound weights with tokenizer.** DistilGPT2 differs from SmolLM2 in training data and parameters as well as segmentation; only the narrow claims are drawn.
- **Model weights live in the shared HF hub cache.** Nothing is copied into the repository; `HF_HUB_OFFLINE=1` guarantees no network access at run time.
