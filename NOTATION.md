# Notation and vocabulary

Every symbol, variable, and named concept used across the studies in this repository, with its full name, its meaning, and where it comes from. Three standing rules:

1. **Capital = batched, lowercase = one position.** `Z` is `(T, V)`; `z` is one row of it.
2. **A subscript that indexes is a coordinate; a subscript that names a knob is a family member.** `z_y` is the true token's logit (coordinate); `p_T` is the temperature-`T` distribution (family member).
3. **`p_1` is reserved** for the untouched model distribution at `T = 1`. Every decoding operator in the later studies is defined as a map `p_1 → p′`, so that name never gets recycled.
4. **`p` is always the model's distribution.** Information-theory texts write `p` for truth and `q` for the model; this repository does not. Where the data-generating distribution is needed it is written **`p_true`**, so `p` never changes meaning between studies. The cross-entropy decomposition therefore reads `E[−log p(x_t)] = H(p_true) + D_KL(p_true ‖ p)`.
5. **Hypothesis-board claims are labelled `C1 … C6`** (Claim), never `H1 … H6`, so that **`H` unambiguously means Shannon entropy** — a measured quantity in several studies. The only other `H` in the repository is `H_final`, the hidden-state matrix, and it appears in prose exactly once.

---

## Core tensors

| Symbol | Code | Full name | What it is |
|---|---|---|---|
| `Z` | `Z` | logit matrix | `(T, V)` float32. One forward pass over the passage. Row `t` scores every vocabulary token as a continuation of the prefix ending at `t`. Unnormalized, unbounded. Mechanically `Z = H_final · W_Eᵀ` — hidden states read against the embedding matrix. |
| `z` | `z = Z[10]` | logit row / logit vector | One position's scores, `R^V`. The single object the whole first study analyses. |
| `z_i` | `z[i]` | logit of token `i` | One coordinate. Only differences `z_i − z_j` are meaningful (softmax is shift-invariant). |
| `z_y` | `Zc[idx, y]` | logit of the true token | The one coordinate cross-entropy reads positively. |
| `Zc` | `Zc = Z[:-1]` | causally-shifted logits | Row `t` predicts token `t+1`, so the last row has no target. `(T−1, V)`. |
| `P` | `P = softmax(Z, -1)` | probability matrix | `(T, V)`, every row a distribution. Capital because batched. |
| `p` | `p = softmax(z, -1)` | probability vector / distribution | `p_i ∈ (0,1)`, `Σ p_i = 1`. A point on the `(V−1)`-simplex. |
| `p_i` | `p[i]` | probability of token `i` | |
| `p_y` | `exp(-loss[i])` | probability of the true token | `L = −log p_y`. |
| `p_1` | `softmax(z, -1)` | base distribution | `p` at `T = 1`. The reference every operator is measured against. |
| `p_T` | `softmax(z/Tt, -1)` | temperature-`T` distribution | `p_T ∝ p_1^{1/T}`. |
| `p_β` | same, `β = 1/T` | inverse-temperature distribution | Used only in the `dH/dT` derivation, where `β` is the natural variable. |
| `p_ref` | `p_ref` | invariance reference | A frozen copy of `p_1` held fixed while the *implementation* is varied. |
| `p_stable`, `p_naive` | | two implementations | Same mathematics, different float32 behavior. Not two distributions — two ways of computing one. |
| `p0` | `softmax(z/0.01, -1)` | zero-temperature limit | One-hot at `argmax z`, in practice. |
| `srt` | `torch.sort(p, descending=True).values` | sorted mass | `p` in descending order. Every truncation operator cuts *this* object, not `p`. |
| `logp` | `torch.log_softmax(Zc, -1)` | log-probabilities | Computed stably. Never `log(softmax(...))` — that reintroduces the underflow the log-space form avoids. |
| `ids` | `ids` | token ids | `(1, N)` integer tensor from the tokenizer. |
| `y` | `y = ids[0, 1:]` | targets | The true next tokens, shifted by one against `Zc`. |
| `idx` | `torch.arange(n)` | row indexer | Pairs with `y` to gather the diagonal `logp[t, y_t]`. |

---

## Scalars and dimensions

| Symbol | Code | Full name | Value / meaning |
|---|---|---|---|
| `T` | `T, V = Z.shape` | **token count** | 83 in study 1. **Overloaded** — see next row. |
| `T` (math) | `Tt` in loops | **temperature** | The logit divisor. Written `Tt` in code precisely because `T` is taken by the token count. |
| `V` | `V` | vocabulary size | 49,152 for SmolLM2. 50,257 for GPT-2 / DistilGPT2. |
| `N` | `N` | number of scored tokens | `T − 1` after the causal shift; in study 2 it varies with the tokenizer. |
| `d` | — | model width / hidden size | 576 for SmolLM2-135M. The vocab/width ratio is 85×. |
| `m` | `z.max()` | row maximum | Subtracted inside every stable softmax. |
| `c` | `c` | shift constant | A scalar added to *every* coordinate: `z → z + c·1`. Not a model parameter — a probe of softmax's redundant degree of freedom. |
| `L` | `L` | evaluation window length | Study 2. How many tokens of context each conditional gets. |
| `S` | `S` | stride | Study 2. Step between successive windows. **`S = L − 1` is disjoint chunking** — a window of `L` tokens has no predictor for its own first token, so `S = L` is arithmetically impossible. `S = 1` scores every token at maximum context. |
| `n_sig`, `m_sig` | | designed head size / mass | Study 3. The oracle split: `n_sig` symbols carrying total mass `m_sig`. |
| `s` | `s_head`, `s_tail` | Zipf exponent | Study 3. `w_i ∝ i^{−s}`. Large `s` = steep and concentrated; small `s` = flat with a fat tail. |
| `k` | `k` | top-k count | Study 3–4. A fixed *count* of surviving symbols. |
| `p_nuc` | `p_nuc` | nucleus mass | Study 3–4. A fixed *mass* threshold; the count adapts. |
| `α` | `alpha` | min-p relative threshold | Study 3–4. Keep `p_i ≥ α·max(p)`; a fixed *relative height*. |
| `M` | `mass_kept` | retained mass | Study 3. `Σ_{i∈S} p_i`. The only quantity `D_KL(p′‖p)` depends on. |
| `B` | `B` | byte count of the raw string | Study 2. The tokenizer-invariant denominator. |
| `Λ` | `total_bits` | total surprise in bits | `Σ_t −log₂ q(x_t \| x_{<t})`. A property of model-plus-tokenizer as a joint compressor. |
| `eps` | `torch.finfo(torch.float32).eps` | machine epsilon | `1.192e-07`. The accuracy floor every identity claim is judged against. |
| `β` | — | inverse temperature | `β = 1/T`. The variable in which `log Z(β)` is a cumulant generating function. |

---

## Metrics

| Symbol | Code | Full name | Definition and reading |
|---|---|---|---|
| `H` | `H` | **Shannon entropy** (nats) | `H(p) = −Σ_i p_i log p_i`. The mean surprise the distribution assigns to its own draws. Range `[0, log V]`. Measures how uncertain the model *says* it is. |
| `log V` | `logV` | uniform-entropy ceiling | `log 49152 = 10.8027` nats. Maximum possible `H`; reached only by the uniform distribution. |
| `exp(H)` | `H.exp()` | **effective support** | The size of the uniform distribution with the same entropy. Reads directly as a token count. **Mass-weighted** — dominated by the head, insensitive to a long thin tail. |
| `top1` | `srt[0]` | leading-token mass | `max_i p_i`. |
| `k90` | `k90` | **effective width at 90% mass** | Smallest `m` with `Σ_{i≤m} srt_i ≥ 0.90`. A raw **count** — must walk the tail. Where `k90 ≫ exp(H)`, the tail is doing the work. |
| `Var_{p_T}(z)` | `(p*z**2).sum() - (p*z).sum()**2` | logit variance under `p_T` | Variance of the **logit values** weighted by the current distribution — *not* the variance of `p`. Drives `dH/dT`. |
| `L`, NLL | `loss_a` | **negative log-likelihood** / cross-entropy | `L(z,y) = −log p_y` in nats. `L = 0` is certainty; `L = log V` is exactly as surprised as a uniform guess; `L > log V` means the model bet against the truth. |
| `PPL` | `ppl` | **perplexity** | `exp(mean NLL)`. The geometric mean of `1/p_y`. Reads as an "effective branching factor". Dominated by the worst positions because it exponentiates an average of logs. |
| `BPB` | `bpb` | **bits per byte** | `Λ / B = (N/B)·log₂(PPL)`. The tokenizer-invariant compression rate. The only honest cross-tokenizer comparison. |
| `N/B` | `fertility` | **fertility** | Tokens per byte. Exactly the factor by which a tokenizer inflates or deflates per-token perplexity. |
| `KL` | `kl` | **Kullback–Leibler divergence** | `D_KL(p′‖p) = Σ p′_i log(p′_i/p_i)`. For any pure truncation this equals `−log(mass_kept)` exactly — it measures *how much* was cut, never *whether the right thing* was cut. The reverse direction `D_KL(p‖p′)` is infinite for any truncation. |
| `p′` | `pp` | truncated distribution | Study 3–4. `p′_i = p_i·1[i∈S]/M`. Zero outside the surviving support. |
| `noise_removed` | | oracle true-positive rate | Study 3. Fraction of the *designed tail mass* excluded by the operator. |
| `signal_lost` | | oracle false-positive rate | Study 3. Fraction of the *designed head mass* excluded. Both are mass fractions, not symbol counts. |

---

## Named operations

| Name | Code | Full name | What it does |
|---|---|---|---|
| softmax | `torch.softmax` | normalized exponential | `p_i = e^{z_i}/Σ_j e^{z_j}`. Maps `R^V` onto the simplex. Invariant to `z + c`, **not** to `a·z`. |
| logsumexp | `torch.logsumexp` | log-sum-exp | `m + log Σ_j e^{z_j − m}`. The log-normalizer. Stable by construction because every exponent argument is `≤ 0`. |
| log_softmax | `torch.log_softmax` | log of softmax, computed in log space | `z_i − logsumexp(z)`. The accurate route to `log p`. |
| cross_entropy | `F.cross_entropy` | fused NLL | Identical kernel to `log_softmax` + gather; agrees with it bit-for-bit (measured 0.0). |
| the LSE trick | — | max-subtraction | Subtract `m = max(z)` before exponentiating. Makes overflow structurally impossible; trades it for harmless underflow. Buys ~141× accuracy even where the naive form survives. |
| catastrophic cancellation | — | — | Subtracting two nearby large numbers to get a small one: leading digits annihilate and the result inherits the operands' *absolute* error. Why `logsumexp(z) − z_y` is 250× less accurate than the gather route. |
| binade | — | — | The interval `[2^k, 2^{k+1})`. Additions that stay inside one binade are exact in floating point; crossing one costs a rounding. Explains the shift-invariance residual pattern (`2.255e-17` at `c=−10` vs `1.192e-07` at `c=±100`). |
| causal shift | `Z[:-1]` vs `ids[1:]` | — | Row `t` predicts token `t+1`. The most common off-by-one bug in perplexity code. |
| power-law tail | — | — | `p_rank ∝ rank^{−α}`; a straight line on log-log axes. No natural cutoff, which is why truncation must be imposed rather than discovered. |

---

## Identities worth memorizing

```
softmax(z + c·1) = softmax(z)                        shift invariance
logsumexp(z + c) = logsumexp(z) + c                  same fact, log space
L(z,y) = −log p_y = logsumexp(z) − z_y               cross-entropy is a lookup
∂L/∂z_i = p_i − 1[i = y]                             the gradient is the distribution
p_T ∝ p_1^{1/T}                                      temperature is a power law
log(p_T,i / p_T,j) = (z_i − z_j)/T                   log-odds scale linearly; argmax is T-invariant
dH/dT = Var_{p_T}(z) / T³  ≥ 0                       entropy is monotone in temperature
PPL = exp(mean NLL) = geometric mean of 1/p_y        perplexity's real definition
BPB = (N/B)·log₂(PPL)                                the fertility correction
E[−log q(x_t | short context)] ≥ E[−log q(x_t | full)]   conditioning cannot hurt on average
```

---

## Numerical constants that recur

| Constant | Value | Where it binds |
|---|---|---|
| float32 max | `3.40e38` | `exp(u) = inf` for `u > log(3.40e38) = 88.72` — the overflow wall. |
| float32 eps | `1.192e-07` | The accuracy floor for every identity claim. |
| float32 underflow | `e^u ≈ 0` for `u < −87.3` | Harmless: those tokens carry `p < 1e-38`. |
| `log V` (SmolLM2) | `10.8027` nats | Uniform-entropy ceiling; `exp(log V) = 49,152`. |
| measured max logit | `33.881` | Leaves **54.84** of headroom to the wall — 38% of the exponent budget already spent. |
| naive-softmax floor | `1.681e-05` | 141× eps, from summing 49,152 terms across 24 orders of magnitude. |
