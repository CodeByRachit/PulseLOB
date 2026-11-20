# PulseLOB: Mathematical Specification

This document specifies the mathematics behind PulseLOB: real-time cross-venue order-flow toxicity detection with an online, non-stationary, marked multivariate Hawkes process.

| Field | Value |
|---|---|
| Spec version | 0.2.0 (changes in [Appendix C](#appendix-c-changes-in-spec-version-020)) |
| Phase | 1 (mathematical research and formulation); revised in Phase 3 |
| Status | Implemented in Phase 3 (`include/pulselob/`); verification results in Section 10 |
| Date | 2026-11-20 |
| Numerical evidence | `research/phase1/validate_math_spec.py` (all claims marked **[Qn]** or **[Fn]** are reproduced by that script) |

---

## 0. Summary of the design

The research question is how to detect cross-venue toxic order flow in real time with a model that adapts online to regime changes, while keeping the per-event cost constant in the length of the event history. This specification answers it with seven design decisions. Each is derived below, with the alternatives that were rejected and why.

1. **Model.** A linear, marked, multivariate Hawkes process on $D$ event channels (base case $D=4$: aggressive buys and sells on a perpetual-futures venue and a spot venue). The excitation kernels are a fixed bank of $Q$ normalized Erlang kernels. Pure exponentials capture fast self-excitation, and Erlang-2 "humps" peaking at 5–20 ms capture the delayed futures-to-spot lead-lag. With the decay rates fixed, the intensity is *linear in the parameters*, $\boldsymbol\lambda(t)=\boldsymbol\Theta(t)\,\mathbf z(t)$. The regressor $\mathbf z(t)$ is parameter-free and shared by all target channels. Every later result builds on this structural fact.
2. **Exact $O(1)$ state.** The Erlang states obey an exact binomial recursion (Theorem 2). Compensators have closed forms in terms of the functions $\psi_n(x)=\int_0^1 s^n e^{-xs}\,ds$ (Theorem 3). These must be evaluated in series-protected form, because the textbook closed form loses 3.5e-9 relative accuracy at sub-microsecond gaps **[Q2]**. The spec also fixes exact semantics for ties, out-of-order data, underflow and timestamp precision (Section 4).
3. **Online estimator (the contribution).** The estimator is a *discounted least-squares contrast* estimator. Its sufficient statistics do not depend on the parameters, so the online estimate equals the batch discounted estimate exactly (Theorem 10), verified to 6.8e-14 against a definition-level rebuild **[Q5]**. The sufficient statistics are maintained with three ingredients: an exact low-rank factorization of each inter-event Gram increment (Theorem 7), square-root (Potter) inverse updates, and a "ridge replenishment" scheme that provably prevents covariance wind-up (Proposition 9). Cost per event is $O(r_{\max} Q^3 D^2)$ time and $O(Q^2D^2)$ memory, independent of the history length $N$ (Theorem 11).
4. **Regime adaptation.** A two-sided CUSUM on time-rescaled residuals (which are i.i.d. Exp(1) under a correct model) switches the forgetting rate from slow to fast when the model stops fitting. In the validation experiment this halves the tracking lag (49.0 s to 22.6 s) with no loss of calm-period accuracy **[F2]**.
5. **Stability guardrail.** Served parameters are projected, exactly and row by row, onto a convex polytope certified sub-critical by the Collatz–Wielandt bound. This gives $\rho(\mathbf G)\le\rho_{\max}<1$ at every event, in $O(D^2Q)$ time (Proposition 14). Theorem 15 then bounds the model's mean intensity uniformly in time, even during a cascade. The size of the unprojected certificate is exposed as a *criticality index*.
6. **Toxicity.** The score $\tau^\pm(t)\in[0,1)$ is the product of two factors. The first is the fraction of forecast impact-weighted flow that is anomalous relative to the model's own stationary mean. The second is how one-sided that anomaly is. The score is directionally exclusive ($\tau^+\tau^-=0$), invariant to two-sided volume, and rises *before* the lagging venue reacts, because it is computed from horizon forecasts rather than instantaneous intensities (Proposition 17).
7. **Control law.** The Avellaneda–Stoikov quotes are re-derived for a jump-diffusion mid-price whose jumps arrive with the Hawkes intensities (Theorem 19). The result has three parts. The reservation price is skewed by the expected *excess* signed impact of the current cascade, $(\mathbf I-\mathbf G)^{-1}\mathbf y^\infty$. Only the endangered side widens, by a Glosten–Milgrom adverse-selection premium $J^b+\tau J^I$. A side is pulled exactly when its break-even distance exceeds the quoting cap. A Schmitt trigger with a cooldown provably bounds message traffic (Theorem 20).

**Why least squares and not online EM.** Online EM was the obvious candidate. On a controlled step change of the cross-venue coupling it was dominated at every lag: it was biased by +0.02 to +0.07 and needed 2.4–3.6 forgetting half-lives to cover half of the step. The least-squares estimator was unbiased (within ±0.002) and needed one half-life, which is the theoretical optimum for a discounted window (Proposition 12). A recursive Newton maximum-likelihood estimator was also dominated **[F1]**. Section 5.8 has the full table.

---

## 1. Notation and conventions

| Symbol | Meaning |
|---|---|
| $D$ | number of event channels, indexed $i,j\in\lbrace 1,\dots,D\rbrace$ |
| $v(i)$, $s_i$ | venue of channel $i$; price direction $s_i\in\lbrace -1,0,+1\rbrace$ of channel $i$ |
| $(t_k,d_k,m_k)$ | $k$-th event: time (seconds, as a double), channel, raw mark features |
| $T_k$ | the same timestamp as a signed 64-bit integer count of nanoseconds |
| $N_i(t)$, $\mathcal F_t$ | counting process of channel $i$; natural filtration of all channels and marks |
| $Q$, $h_q$, $\beta_q$, $r_q$ | kernel bank size; normalized Erlang kernel $q$ with rate $\beta_q$ and order $r_q$ |
| $P=1+DQ$ | number of parameters per target channel |
| $M+1$ | number of time-basis functions, $M=\sum_q r_q$ |
| $\mathbf z(t)\in\mathbb R^P$ | parameter-free regressor: a 1 followed by the kernel features $x^q_j(t)$ |
| $\boldsymbol\Theta\in\mathbb R^{D\times P}$ | parameters; row $i$ is $\theta_i=(\mu_i,\alpha^q_{ij})$ |
| $\tilde{\boldsymbol\Theta}$, $\hat{\boldsymbol\Theta}$ | unconstrained online estimate; served (projected, feasible) estimate |
| $\mathbf G$ | branching matrix, $G_{ij}=\sum_q\alpha^q_{ij}$ |
| $\kappa(t)$, $\mathcal K(s,t)$ | forgetting rate; integrated forgetting $\int_s^t\kappa(u)\,du$ |
| $\rho(\cdot)$ | spectral radius |

Conventions:

* Vector and matrix inequalities are componentwise. $(x)_+=\max(x,0)$. $0/0:=0$ wherever a ratio of nonnegative quantities appears.
* All intensities are left-continuous: $\lambda_i(t)$ depends on events with $t_k<t$ (strict inequality). The value immediately after a time $t$ is written $t+$.
* Time differences are computed in integer nanoseconds and only then converted to seconds: $\Delta_k=10^{-9}\,(T_k-T_{k-1})$ (Section 4.7).
* Rates are per second. Prices are in quote currency, or in ticks where stated.

---

## 2. Event channels and marks

### 2.1 Channels

The base model has four channels. A perpetual-futures venue A (for example Binance USD-M) leads, and a spot venue B (for example Binance Spot) lags.

| $i$ | Channel | Venue | $s_i$ |
|---|---|---|---|
| 1 | aggressive buy | A (perp) | $+1$ |
| 2 | aggressive sell | A (perp) | $-1$ |
| 3 | aggressive buy | B (spot) | $+1$ |
| 4 | aggressive sell | B (spot) | $-1$ |

The specification is generic in $D$. The planned extensions are:

* **Top-of-book depletion** channels, one per side per venue ($D=8$). An event fires when the best-level queue falls by more than a fraction $f_{\mathrm{dep}}$ of its EWMA size *without* a matching trade, which identifies a cancellation-driven depletion. Bid depletion is bearish ($s_i=-1$) and ask depletion bullish ($s_i=+1$).
* **OFI shock** channels ($D$ up to 12). With the order-flow imbalance of Cont, Kukanov and Stoikov (2014) accumulated since the last shock, an event fires each time it crosses $\pm k_{\mathrm{ofi}}\,\hat\sigma_{\mathrm{ofi}}$. The sign of the crossing gives $s_i$.

Nothing below depends on $D=4$, but the cost formulas in Section 5.6 depend on $D$ explicitly.

### 2.2 Event construction

* **Aggressor side** comes from the exchange flag, so no Lee–Ready inference is needed. On Binance trade streams, `isBuyerMaker = true` means the aggressor was a *seller*.
* **Sweep aggregation.** Prints with the same (venue, aggressor side, exchange timestamp) are merged into one event. Its size is the summed quantity, and its sweep depth is $L=\lvert p_{\mathrm{last}}-p_{\mathrm{first}}\rvert/\text{tick}$. This removes the most common source of exact ties (Section 4.4) and matches the economic unit, which is one aggressive order.
* **Ordering.** The engine consumes a single merged stream in non-decreasing timestamp order. Phase 4 defines the clock-alignment and reorder-buffer policy. The engine itself only has to tolerate the residual disorder described in Section 4.5.

### 2.3 Mark function

Each event carries three raw features:

* $u_k=\text{size}_k/\widehat{\mathrm{med}}_{d_k}$: size relative to a running median of the same channel;
* $L_k\ge 0$: sweep depth in ticks;
* $\iota_k=s_{d_k}\cdot\mathrm{OBI}_k\in[-1,1]$: the top-of-book imbalance $\mathrm{OBI}=(q^{\mathrm{bid}}-q^{\mathrm{ask}})/(q^{\mathrm{bid}}+q^{\mathrm{ask}})$ just before the event, aligned with the trade's direction. $\iota>0$ means a buy into a thin ask, or a sell into a thin bid, which is the configuration in which an aggressive order is most likely to move the price.

The raw mark is

```math
\tilde g(m_k) \;=\; \bigl[1+c_u\ln\!\bigl(1+\min(u_k,u_{\max})\bigr)\bigr]\cdot
\bigl[1+c_L\ln\!\bigl(1+\min(L_k,L_{\max})\bigr)\bigr]\cdot
\bigl[1+c_\iota\,\iota_k\bigr],
\qquad c_u,c_L\ge 0,\; c_\iota\in[0,1).
```

**Lemma 1 (bounded marks).** For every input, $\tilde g\in[\tilde g_{\min},\tilde g_{\max}]$ with $\tilde g_{\min}=1-c_\iota>0$ and $\tilde g_{\max}=(1+c_u\ln(1+u_{\max}))(1+c_L\ln(1+L_{\max}))(1+c_\iota)$.

*Proof.* The first two factors are at least 1 because $\ln(1+x)\ge 0$ for $x\ge0$, and they are increasing and capped by the $\min$. The third factor lies in $[1-c_\iota,1+c_\iota]$ because $\lvert\iota\rvert\le1$. $\square$

With the defaults $c_u=c_L=c_\iota=0.5$, $u_{\max}=10^4$ and $L_{\max}=50$, this gives $\tilde g\in[0.5,\,24.94]$. Log-concave growth means a 10,000× outlier print excites the model only about 5.6× more than a median print, so outlier prints cannot overflow anything.

**Online normalization.** The mark actually used is $g_k=\tilde g_k/\bar g_{d_k}$, where $\bar g_j$ is an event-time EWMA of $\tilde g$ on channel $j$ with weight $\eta_g$, clamped to $[\tilde g_{\min},\tilde g_{\max}]$. It is read *before* being updated with event $k$, which keeps it causal. Hence $g_k\in[\tilde g_{\min}/\tilde g_{\max},\,\tilde g_{\max}/\tilde g_{\min}]$ and $\mathbb E[g]\approx 1$. Unit-mean marks make $\alpha^q_{ij}$ directly interpretable as an expected number of children (Section 3.2).

**Running median in $O(1)$.** The median uses a sign-based Robbins–Monro tracker in log space, $\nu_{j}\leftarrow\nu_j+\eta_\nu\,(\mathbb 1\lbrace\ell_k>\nu_j\rbrace-\mathbb 1\lbrace\ell_k<\nu_j\rbrace)$ with $\ell_k=\ln\text{size}_k$, and $\widehat{\mathrm{med}}_j=e^{\nu_j}$. Its fixed point is the median of $\ell$, and it fluctuates in an $O(\eta_\nu)$ band (Robbins and Monro, 1951). Working in log space makes the step multiplicative and therefore scale-free.

**Design trade-off: fixed mark coefficients.** The coefficients $(c_u,c_L,c_\iota)$ are calibrated offline and held fixed online. Learning them online would make the intensity *bilinear* in $(\alpha,c)$. That destroys the linear-in-parameters structure on which exactness, convexity and the $O(1)$ estimator rest. A slow outer loop, for example a daily profile-likelihood refit, is the right place to adapt them.

---

## 3. Intensity model

### 3.1 Definition

```math
\lambda_i(t)=\mu_i(t)+\sum_{j=1}^{D}\sum_{q=1}^{Q}\alpha^q_{ij}(t)\,x^q_j(t),
\qquad
x^q_j(t)=\sum_{k:\,d_k=j,\;t_k<t} g_k\,h_q(t-t_k),
```

```math
h_q(s)=\frac{\beta_q^{\,r_q}\,s^{\,r_q-1}}{(r_q-1)!}\,e^{-\beta_q s}\,\mathbb 1\lbrace s\ge 0\rbrace,
\qquad \int_0^\infty h_q=1,\quad \text{mode}=\tfrac{r_q-1}{\beta_q},\quad\text{mean}=\tfrac{r_q}{\beta_q}.
```

The default kernel bank ($Q=4$) is:

| $q$ | Order $r_q$ | $\beta_q$ (1/s) | Mode | Mean | Purpose |
|---|---|---|---|---|---|
| 1 | 1 | 1000 | 0 | 1 ms | fast self-excitation (order splitting, sweeps) |
| 2 | 2 | 200 | 5 ms | 10 ms | short lead-lag (co-located arbitrage) |
| 3 | 2 | 50 | 20 ms | 40 ms | long lead-lag (cross-region propagation) |
| 4 | 1 | 1 | 0 | 1 s | slow clustering (meta-order persistence) |

For the default bank $P=17$ and $M+1=7$.

Every basis is available to every channel pair, and the estimator decides the weights. A cross-venue pair with a 12 ms lag loads on $q=2$ and $q=3$; a self-excitation pair loads on $q=1$ and $q=4$. Mixing Erlang-2 humps at two modes lets the model represent any lag profile in their span. It does not need a delay parameter, which would break both linearity and $O(1)$ state. (A hard delay $\mathbb 1\lbrace s\ge\delta\rbrace e^{-\beta(s-\delta)}$ needs the events inside the delay window, and that buffer is unbounded.)

### 3.2 Linear-in-parameters form

Define $\mathbf z(t)=(1,\,x^1_1(t),\dots,x^Q_1(t),\,\dots,\,x^Q_D(t))^\top\in\mathbb R^P$ and $\theta_i=(\mu_i,\alpha^1_{i1},\dots,\alpha^Q_{iD})$. Then

```math
\boldsymbol\lambda(t)=\boldsymbol\Theta(t)\,\mathbf z(t),\qquad \boldsymbol\Theta\in\mathbb R^{D\times P}.
```

Three structural facts are used throughout.

* **(F1) Parameter-free regressor.** $\mathbf z(t)$ depends only on event times, channels and marks. It does not depend on $\boldsymbol\Theta$.
* **(F2) Shared regressor.** All $D$ target channels use the same $\mathbf z(t)$.
* **(F3) Linearity.** $\lambda_i$ is linear in $\theta_i$, so the log-likelihood is concave and the least-squares contrast is quadratic.

**Branching matrix.** With unit-mean marks, the expected number of direct children of type $i$ produced by one type-$j$ event is $G_{ij}=\sum_q\alpha^q_{ij}$. The model is stationary if and only if $\rho(\mathbf G)<1$ (Hawkes, 1971; Brémaud and Massoulié, 1996). If marks have conditional mean $\bar g\ne1$, the relevant matrix is $\bar g\,\mathbf G$ (Section 6.5).

**Convention for time-varying parameters.** $\lambda_i(t)$ uses $\boldsymbol\Theta(t)$, the parameters *at evaluation time*, applied to the whole history. This is the convention that makes (F1) hold, and with it the exactness theorem of Section 5. The alternative, freezing each event's contribution at its arrival-time parameters, needs $D^2Q$ parameter-weighted states and breaks the sufficient-statistic structure. Because parameters move on a scale of seconds while kernels decay on a scale of milliseconds, the two conventions differ by $O(\lVert\dot{\boldsymbol\Theta}\rVert\cdot\text{kernel mean})$, which is negligible except for the 1 s kernel.

**Why fixed decay rates.** The rates $\beta_q$ enter nonlinearly, and the likelihood is not concave in them. Fixing a logarithmically spaced bank turns the kernel into a basis expansion, which keeps every estimation problem convex. Re-selecting the bank (for example by profile likelihood) is an offline, slow-timescale task for Phase 5.

**Identifiability conditions.** These are enforced at configuration time:

* (I1) The bank modes and means are distinct.
* (I2) $\kappa_{\max}<\beta_{\min}$. This is also needed for numerical reasons (Section 4.2).
* (I3) The slowest kernel mean is shorter than the slow forgetting half-life by at least 10×. Otherwise the baseline and the slow kernel are confounded within one window. During fast-forgetting episodes this ratio can drop to about 4. There the ridge prior of Section 5.4 provides the missing regularization.

---

## 4. Exact $O(1)$ recursions and numerical safety

### 4.1 Erlang state recursion

For each source channel $j$, basis $q$ and $n=0,\dots,r_q-1$, define

```math
R^{q,(n)}_j(t)=\sum_{k:\,d_k=j,\;t_k<t} g_k\,\frac{(t-t_k)^n}{n!}\,e^{-\beta_q(t-t_k)},
\qquad\text{so that}\qquad x^q_j(t)=\beta_q^{\,r_q}\,R^{q,(r_q-1)}_j(t).
```

**Theorem 2 (exact recursion).**

(a) *Decay.* If no event occurs in $(t,t+\Delta)$, with $\Delta\ge0$, then

```math
R^{q,(n)}_j(t+\Delta)=e^{-\beta_q\Delta}\sum_{l=0}^{n}\frac{\Delta^{\,n-l}}{(n-l)!}\,R^{q,(l)}_j(t+).
```

(b) *Jump.* At an event of channel $j$ at time $t_k$ with mark $g_k$: $R^{q,(0)}_j(t_k+)=R^{q,(0)}_j(t_k)+g_k$, and $R^{q,(n)}_j$ is unchanged for $n\ge1$.

*Proof.* (a) For each past event, $\frac{(t+\Delta-t_k)^n}{n!}=\sum_{l=0}^n\frac{\Delta^{n-l}}{(n-l)!}\frac{(t-t_k)^l}{l!}$ by the binomial theorem, and $e^{-\beta(t+\Delta-t_k)}=e^{-\beta\Delta}e^{-\beta(t-t_k)}$. Summing over $k$ gives the claim. (b) A new event at $t_k$ contributes $g_k\cdot 0^n/n!$, which is $g_k$ for $n=0$ and $0$ for $n\ge1$. $\square$

The two cases used by default are:

```math
\text{Erlang-1: } R^{(0)}\leftarrow e^{-\beta\Delta}R^{(0)};\qquad
\text{Erlang-2: } R^{(1)}\leftarrow e^{-\beta\Delta}\bigl(R^{(1)}+\Delta R^{(0)}\bigr),\;\;R^{(0)}\leftarrow e^{-\beta\Delta}R^{(0)}.
```

In Erlang-2 the update of $R^{(1)}$ uses the *old* $R^{(0)}$.

The recursion matches the $O(N^2)$ definition to 5.6e-16 for orders 1, 2 and 3 over 500 events with ties **[Q1]**.

**Cost consequence.** The decay factor $e_q=e^{-\beta_q\Delta}$ is shared by all source channels. An event therefore needs exactly $Q$ exponentials for the kernel bank, plus one for forgetting, whatever the value of $D$. Pair-specific decays $\beta_{ij}$ would need $D^2Q$ states and $D^2Q$ exponentials, and the shared bank avoids both.

### 4.2 The $\psi$ functions

```math
\psi_n(x)=\int_0^1 s^n e^{-xs}\,ds,\qquad
\psi_0(x)=\frac{1-e^{-x}}{x},\qquad
\psi_n(x)=\frac{n\,\psi_{n-1}(x)-e^{-x}}{x},
\qquad
\int_0^{\Delta}u^n e^{-au}\,du=\Delta^{n+1}\psi_n(a\Delta).
```

**Evaluation rule.** For $\lvert x\rvert<1$ use the series $\psi_n(x)=\sum_{m\ge0}\frac{(-x)^m}{m!\,(n+m+1)}$, truncated at 22 terms, where the truncation error is below $1/22!\approx 9\times10^{-22}$. Otherwise use the upward recurrence. Its error amplification is $n/x\le n$ per step for $x\ge1$, and $n\le2$ in this design.

Measured accuracy is 5.2e-16 relative for $n\le3$ on $x\in[10^{-12},200]$ **[Q3]**. A narrower series radius of $1/4$ was tried first and failed (3.1e-13 at $n=3$, $x=0.2501$), which is why the radius is 1.

The only negative argument in the formulas is $x=-\kappa\Delta$ with $n=0$ (Section 5.4); every other argument is positive by (I2). The engine evaluates that one entry in the equivalent form $d\,\Delta\,\psi_0(-\kappa\Delta)=\Delta\,\psi_0(\kappa\Delta)$, which never forms $e^{+\kappa\Delta}$ and therefore cannot overflow on idle gaps (spec 0.2.0).

**No new exponentials.** Every $e^{-x}$ needed below is a product or ratio of the $Q+1$ factors $e_q$ and $d=e^{-\kappa\Delta}$ already computed for the event.

### 4.3 Compensator over an event-free interval

**Theorem 3.** For $u\in[0,\Delta)$ with no event in $(t,t+\Delta)$,

```math
x^q_j(t+u)=\beta_q^{\,r_q}e^{-\beta_q u}\sum_{l=0}^{r_q-1}\frac{u^{\,r_q-1-l}}{(r_q-1-l)!}\,R^{q,(l)}_j(t+),
```

```math
X^q_j(t,\Delta):=\int_0^\Delta x^q_j(t+u)\,du
=\beta_q^{\,r_q}\sum_{l=0}^{r_q-1}\frac{\Delta^{\,r_q-l}}{(r_q-1-l)!}\,\psi_{r_q-1-l}(\beta_q\Delta)\,R^{q,(l)}_j(t+).
```

The compensator increment of channel $i$ is $\Lambda_i(t,t+\Delta)=\theta_i^\top\mathbf Z(t,\Delta)$, where $\mathbf Z=(\Delta,\,X^1_1,\dots,X^Q_D)$.

The default cases are $X=R^{(0)}\,\beta\Delta\,\psi_0(\beta\Delta)$ for Erlang-1 and $X=\beta^2\bigl(R^{(1)}\Delta\,\psi_0(\beta\Delta)+R^{(0)}\Delta^2\psi_1(\beta\Delta)\bigr)$ for Erlang-2.

*Proof.* The first identity is Theorem 2(a) with $\Delta=u$, multiplied by $\beta^r$. Integrate term by term using the identity in Section 4.2. $\square$

**Mandatory form.** The algebraically equivalent expression $\beta R^{(1)}(1-e^{-x})+R^{(0)}(1-e^{-x}(1+x))$ subtracts nearly equal numbers. Its worst relative error was 3.5e-9 at $\Delta=10^{-9}$ s, which violates the 1e-10 requirement. The $\psi$-form reaches 2.2e-16 **[Q2]**.

### 4.4 Simultaneous timestamps ($\Delta=0$)

**Definition (batch semantics).** All events with the same timestamp form a *batch*. Every event in a batch is evaluated against the pre-batch state ($t-$). Their jumps accumulate in a pending buffer $\boldsymbol\pi\in\mathbb R^D$ ($\pi_{d_k}\mathrel{+}=g_k$), which is flushed into $R^{q,(0)}$ when the first event with a strictly later timestamp arrives (Section 9).

**Proposition 4.**

1. Batch semantics is exactly the strict-inequality convention $t_k<t$: events at the same instant do not excite each other.
2. Every quantity computed for a batch is invariant to the order of events within it. Feed-handler nondeterminism therefore cannot change results.
3. The extra state is $D$ numbers, and a flush costs $O(DQ)$.

*Proof.* (1) is the definition. (2) The pre-batch state is common to all events in the batch, and the pending sum is commutative. (3) is immediate. $\square$

Two further rules apply to batches.

* **Two clocks.** Likelihood, residual and estimator quantities use $t_k-$. Forecasts and trading decisions after event $k$ use the post-event state $R+\boldsymbol\pi$ (read-only, without flushing), because a decision should use all information available at $t_k$.
* **Caveat.** A simple point process has no ties, so a likelihood containing ties is not a density. Ties are reduced upstream by sweep aggregation (Section 2.2). The few that remain are cross-channel coincidences at the timestamp resolution, and batch semantics is the standard deterministic treatment. Random dithering was rejected because it breaks reproducibility.

### 4.5 Out-of-order timestamps ($\Delta<0$)

**Policy.** Let $T_{\mathrm{last}}$ be the engine clock (the largest accepted timestamp). For an event with $T_k<T_{\mathrm{last}}$:

* if $T_{\mathrm{last}}-T_k\le\varepsilon_{\mathrm{ooo}}$, clamp $T_k\leftarrow T_{\mathrm{last}}$. The event joins the current batch, and a counter is incremented.
* otherwise, reject the event without changing any state and increment a counter. A rejection rate above a threshold raises a feed-health alarm (Section 8.6).

**Proposition 5.** Under this policy the engine clock is non-decreasing. Every decay factor lies in $(0,1]$. All $R$ states, the moment matrix and the ridge remain nonnegative or positive definite as before. The engine never evaluates $e^{+\beta\lvert\Delta\rvert}$, which would amplify the state and could overflow.

*Proof.* Clamping or rejecting gives $\Delta\ge0$ for every accepted event. All the invariants of Theorem 2 and of Section 5 are preserved under $\Delta\ge0$. $\square$

The cost of clamping is that the event is evaluated at $t_{\mathrm{last}}-$ instead of $t_k-$, so it "sees" the events in $(t_k,t_{\mathrm{last}})$. That error is bounded by $\varepsilon_{\mathrm{ooo}}\cdot\max_i\lvert\dot\lambda_i\rvert$ plus the excitation from those events. The primary defense is the Phase 4 reorder buffer; $\varepsilon_{\mathrm{ooo}}$ only absorbs residual jitter (default 100 µs).

### 4.6 Underflow, denormals and overflow

* **Cutoff.** If $\beta_q\Delta>X_{\mathrm{cut}}=50$, set $e_q=0$ and zero all states of basis $q$. The neglected mass relative to the state is at most $\max_{n<r}\frac{X^n}{n!}e^{-X}\le\frac{50^2}{2}e^{-50}\approx 2.4\times10^{-19}$ for $r\le3$, which is below double-precision resolution.
* **Denormals.** The C++ engine sets FTZ/DAZ (flush-to-zero, denormals-are-zero). With the cutoff in place, denormal states can only come from repeated decay of already tiny values, and FTZ removes those. This avoids the roughly 100× penalty for denormal arithmetic on x86.
* **Overflow is impossible.** Every state is bounded by $g_{\max}$ times the number of events in the last $X_{\mathrm{cut}}/\beta_q$ seconds, which is finite and far below $10^{308}$. Section 5.4 bounds the moment matrix inverse, and the served parameters lie in a compact set (Section 6).

### 4.7 Timestamp precision

Absolute epoch timestamps in nanoseconds ($\approx1.7\times10^{18}$) exceed $2^{53}\approx9\times10^{15}$. Converting them to double would quantize time to 256 ns. The engine therefore stores timestamps as signed 64-bit integers, subtracts in integers, and converts only the difference to double. A difference is exact up to $2^{53}$ ns, about 104 days.

### 4.8 Rounding-error accumulation

Each update $R\leftarrow eR+g$ with $e\in(0,1]$ is a contraction. With unit roundoff $u$, the relative error obeys $\epsilon_k\le e_k\epsilon_{k-1}+2u$. That makes it at most $2u$ times the number of events within the kernel's effective memory, $1/(1-\bar e)$. Even at $10^5$ events per second on the 1 s kernel, this is $2.2\times10^{-11}$, which is below the 1e-10 test tolerance. If a larger bank or rate requires it, a compensated (Kahan) update is a configuration option.

---

## 5. Online non-stationary estimation (the research contribution)

### 5.1 Requirements

* **R1 Causal:** the estimate at $t$ uses only $\mathcal F_t$.
* **R2 Constant-time:** the per-event cost is independent of $N$ and is $O(D^2)$ for a fixed kernel bank.
* **R3 Zero allocation:** state has a fixed size known at compile time.
* **R4 Tracking:** a regime change is followed at a rate set by the forgetting horizon, without hand-tuned step sizes.
* **R5 Feasibility:** served parameters are positive and certified sub-critical at every event.
* **R6 Verifiability:** the recursive estimate has a brute-force reference it must match to 1e-10.

### 5.2 Candidate estimators

| Method | Objective | Cost per event | Step size | Exactly equal to batch | Positivity | Tracking **[F1]** |
|---|---|---|---|---|---|---|
| (a) Projected online gradient ascent | log-likelihood | $O(D^2Q)$ | needs tuning; $1/\lambda$ makes the gradient unbounded | no | by projection | not run: step-size dependent |
| (b) Recursive Newton ML (online-Newton type) | log-likelihood | $O(P^2)=O(D^2Q^2)$, with a lazy per-channel update | implicit | no (stale Hessian, $\theta$-dependent score) | clamp | lag ≈ 2× half-life |
| (c) Online EM, discounted sufficient statistics (Cappé and Moulines, 2009) | likelihood via latent branching structure | $O(DQ)$ update + $O(D^2Q)$ evaluation | none | no (credit assigned with stale parameters) | automatic | biased; lag ≈ 2.4–3.6× half-life |
| **(d) Discounted least-squares contrast (chosen)** | quadratic contrast | $O(r_{\max}Q^3D^2)$ | **none** | **yes (Theorem 10)** | by exact projection (Section 6) | **unbiased; lag ≈ 1× half-life** |
| (e) Iteratively reweighted least squares | approximates the likelihood | $O(DP^2)=O(D^3Q^2)$ | none | no | projection | not run; **violates R2** |

**Why online EM lags.** Online EM attributes each event to its possible parents with the parameters current at its arrival, and never revisits that attribution. After a jump in the cross-venue coupling, the excess events are credited mostly to the baseline and to self-excitation, in proportion to the *old* coupling. The coupling estimate therefore moves at the forgetting rate multiplied by EM's linear convergence rate, which is the fraction of missing information and is large for Hawkes processes. That is the measured 2.4–3.6× lag. Its finite-window bias (+0.02 to +0.07 at a true value of 0.10) comes from the same stale attribution.

**Why least squares is not the textbook-optimal choice, and why it wins here anyway.** The likelihood is statistically more efficient (Proposition 12(c)). In practice, though, the sufficient statistics of every likelihood-based online method depend on the parameters. They are therefore computed with stale parameters, which costs bias and lag. The least-squares contrast is the only candidate whose sufficient statistics are parameter-free, by (F1). Its online estimate is therefore *identical* to the batch estimate. That one property delivers R4, R6, the absence of step sizes, and the near-ideal lag measured in **[F1]**.

### 5.3 The discounted least-squares contrast

For target channel $i$, with discount weight $w_t(s)=e^{-\mathcal K(s,t)}$, a diagonal ridge matrix $\boldsymbol\Omega_t$ and a prior mean $\theta_{0,i}$ taken from an offline static calibration:

```math
\mathcal C_t(\theta_i)=\int_0^t w_t(s)\,\lambda_i(s;\theta_i)^2\,ds-2\int_0^t w_t(s)\,\lambda_i(s-;\theta_i)\,dN_i(s)+(\theta_i-\theta_{0,i})^\top\boldsymbol\Omega_t(\theta_i-\theta_{0,i}).
```

This is the least-squares contrast for point processes, $-2\int f\,dN+\int f^2dt$ (Hansen, Reynaud-Bouret and Rivoirard, 2015; Bacry et al., 2020), with exponential discounting and a Gaussian prior added. Using (F3), $\lambda_i=\theta_i^\top\mathbf z$, so

```math
\mathcal C_t(\theta_i)=\theta_i^\top\mathbf M_t\,\theta_i-2\,\theta_i^\top\mathbf c_{i,t}+\text{const},\qquad
\mathbf M_t=\int_0^t w_t(s)\,\mathbf z(s)\mathbf z(s)^\top ds+\boldsymbol\Omega_t,\qquad
\mathbf c_{i,t}=\int_0^t w_t(s)\,\mathbf z(s-)\,dN_i(s)+\boldsymbol\Omega_t\theta_{0,i}.
```

Stacking the targets gives the unique minimizer

```math
\boxed{\;\tilde{\boldsymbol\Theta}_t=\mathbf C_t\,\mathbf M_t^{-1}\;},\qquad \mathbf C_t=[\mathbf c_{1,t},\dots,\mathbf c_{D,t}]^\top\in\mathbb R^{D\times P}.
```

By (F2) there is one $P\times P$ moment matrix shared by all $D$ targets. This is what makes an $O(D^2)$ second-order method possible.

**Theorem 6 (unbiased estimating equation).** Suppose that on the window the true intensity is $\lambda^\star_i=\theta^{\star\top}_i\mathbf z$ with $\theta^\star_i$ constant, the process has finite second moments, and $\boldsymbol\Omega=0$. Then $\mathbb E[\mathbf c_{i,t}-\mathbf M_t\theta^\star_i]=0$.

*Proof.* $\mathbf c_{i,t}-\mathbf M_t\theta^\star_i=\int_0^t w_t(s)\,\mathbf z(s-)\,\bigl(dN_i(s)-\lambda^\star_i(s)\,ds\bigr)$. The integrand $w_t(s)\mathbf z(s-)$ is predictable: $\mathbf z$ is left-continuous, and $\kappa$ on each inter-event interval is fixed at its left endpoint (Section 5.5). The compensated process $N_i-\int\lambda^\star_i$ is a martingale, so the stochastic integral has mean zero (Daley and Vere-Jones, 2003). $\square$

### 5.4 Exact recursive sufficient statistics

The recursion maintains five things:

* the inverse square-root factor $\mathbf U$, with $\mathbf U\mathbf U^\top=\mathbf M^{-1}$ ($P\times P$);
* $\tilde{\boldsymbol\Theta}$ and $\mathbf C$ (each $D\times P$);
* the effective ridge $\boldsymbol\omega^{\mathrm{eff}}\in\mathbb R^P$;
* the $R$ states and the pending buffer $\boldsymbol\pi$;
* the replenishment counter.

**Theorem 7 (exact low-rank interval increment).** On an event-free interval $(t,t+\Delta)$, the regressor has the form $\mathbf z(t+u)=\mathbf V^\top\mathbf f(u)$. Here $\mathbf f(u)=(f_0,\dots,f_M)$ with $f_m(u)=u^{n_m}e^{-c_m u}$: $f_0\equiv1$, and for each basis $q$ and $n<r_q$ there is a function $u^ne^{-\beta_q u}$. $\mathbf V\in\mathbb R^{(M+1)\times P}$ holds the coefficients from Theorem 3: coordinate 0 has coefficient 1 on $f_0$, and coordinate $(j,q)$ has coefficient $\beta_q^{r_q}R^{q,(r_q-1-n)}_j(t+)/n!$ on $u^ne^{-\beta_qu}$. Then

```math
\mathbf M_{t+\Delta}=d\,\mathbf M_t+\mathbf V^\top\mathbf F\,\mathbf V+(\text{ridge bookkeeping}),\qquad
F_{ab}=d\,\Delta^{\,n_a+n_b+1}\,\psi_{n_a+n_b}\bigl((c_a+c_b-\kappa)\Delta\bigr),\quad d=e^{-\kappa\Delta}.
```

$\mathbf F$ is positive semi-definite with rank at most $M+1$. If $\mathbf F=\sum_r\boldsymbol\ell_r\boldsymbol\ell_r^\top$ is its pivoted Cholesky factorization, the increment is $\sum_r\mathbf w_r\mathbf w_r^\top$ with $\mathbf w_r=\mathbf V^\top\boldsymbol\ell_r$. That is at most $M+1$ rank-one updates, and $M+1$ does not depend on $D$.

*Proof.* Substitute $\mathbf z=\mathbf V^\top\mathbf f$ into $\int_0^\Delta e^{-\kappa(\Delta-u)}\mathbf z\mathbf z^\top du=\mathbf V^\top\bigl(\int_0^\Delta e^{-\kappa(\Delta-u)}\mathbf f\mathbf f^\top du\bigr)\mathbf V$. Each entry of the inner integral is $d\int_0^\Delta u^{n_a+n_b}e^{-(c_a+c_b-\kappa)u}du$, which Section 4.2 evaluates. $\mathbf F$ is the Gram matrix of the functions $f_m$ in a weighted $L^2$ space, so it is positive semi-definite. $\square$

Additional notes on Theorem 7:

* **Arguments and exponentials.** $c_a+c_b\in\lbrace 0,\beta_q,\beta_q+\beta_{q'}\rbrace$, and $e^{-(c_a+c_b-\kappa)\Delta}=e_qe_{q'}/d$, so no new exponentials are needed. The argument is negative only when $c_a=c_b=0$, which gives $n=0$ (Section 4.2).
* **Truncation.** Pivoting stops when the largest remaining diagonal entry is at most $\eta\operatorname{tr}\mathbf F$. The remaining Schur complement is positive semi-definite with trace at most $(M+1)\eta\operatorname{tr}\mathbf F$, so the dropped part has spectral norm at most $(M+1)\eta\operatorname{tr}(\mathbf F)\lVert\mathbf V\rVert_2^2$. The default $\eta=10^{-15}$ puts this at rounding level.
* **Measured.** The factorized increment matches adaptive quadrature to 5.2e-16. Its numerical rank adapts to the interval length: rank 2 at $\Delta=10^{-7}$ s, rising to 5 (the full $M+1=5$ in the test bank) at $\Delta\ge0.02$ s. Bursts with tiny gaps therefore cost fewer updates **[Q4]**.

**Event increment.** At an event of channel $i$: $\mathbf C_{i,:}\mathrel{+}=\mathbf z(t_k-)^\top$ and $\mathbf M$ is unchanged. Hence $\tilde\theta_i\mathrel{+}=(\mathbf U\mathbf U^\top\mathbf z)^\top$, at cost $O(P^2)$.

**Lemma 8 (square-root rank-one update; Potter, see Bierman, 1977).** Let $\mathbf U\mathbf U^\top=\mathbf M^{-1}$, $\mathbf a=\mathbf U^\top\mathbf w$, $\sigma^2=\mathbf a^\top\mathbf a$ and $\varsigma=1/\bigl(\sqrt{1+\sigma^2}\,(\sqrt{1+\sigma^2}+1)\bigr)$. Then

```math
\mathbf U'=\mathbf U-\varsigma\,(\mathbf U\mathbf a)\,\mathbf a^\top\ \Rightarrow\ \mathbf U'\mathbf U'^\top=(\mathbf M+\mathbf w\mathbf w^\top)^{-1},\qquad
\mathbf k:=(\mathbf M+\mathbf w\mathbf w^\top)^{-1}\mathbf w=\frac{\mathbf U\mathbf a}{1+\sigma^2},\qquad
\tilde{\boldsymbol\Theta}'=\tilde{\boldsymbol\Theta}-(\tilde{\boldsymbol\Theta}\mathbf w)\,\mathbf k^\top.
```

*Proof.* $(\mathbf I-\varsigma\mathbf a\mathbf a^\top)^2=\mathbf I-(2\varsigma-\varsigma^2\sigma^2)\mathbf a\mathbf a^\top$. The stated $\varsigma$ is the stable rewriting of $(1-(1+\sigma^2)^{-1/2})/\sigma^2$, and it satisfies $2\varsigma-\varsigma^2\sigma^2=(1+\sigma^2)^{-1}$. That gives Sherman–Morrison in factored form. The update of $\tilde{\boldsymbol\Theta}$ follows from $\mathbf C(\mathbf M+\mathbf w\mathbf w^\top)^{-1}=\tilde{\boldsymbol\Theta}-(\tilde{\boldsymbol\Theta}\mathbf w)\mathbf k^\top$. $\square$

Forgetting by $d$ maps $\mathbf U\leftarrow\mathbf U/\sqrt d$ and $\mathbf C\leftarrow d\,\mathbf C$, and leaves $\tilde{\boldsymbol\Theta}$ unchanged. Cost is $O(P^2+DP)$ per rank-one update.

**Measured.** After $10^5$ ill-conditioned updates ($\operatorname{cond}\mathbf M=10^{12}$), the relative error was 8.0e-13 for the Potter factor and 4.5e-15 for plain Sherman–Morrison with periodic symmetrization **[Q6]**. Potter is chosen for its *structural* guarantee: $\mathbf U\mathbf U^\top$ is positive semi-definite by construction, so the gain can never change sign, whatever the rounding (Verhaegen, 1989). The cheaper Sherman–Morrison variant (about 2/3 of the flops) remains a benchmarked alternative for Phase 3.

**Ridge replenishment (prevents wind-up).** With pure exponential forgetting, a channel that stops firing makes $\mathbf M$ singular, and $\mathbf M^{-1}$ grows without bound. This is the classical wind-up problem (Kulhavý and Zarrop, 1993). A constant ridge $\boldsymbol\Omega$ would fix it, but its exact maintenance needs the full-rank term $(1-d)\boldsymbol\Omega$ at every interval. Replenishment approximates it with bounded error instead.

The ridge decays with everything else ($\omega^{\mathrm{eff}}_p\leftarrow d\,\omega^{\mathrm{eff}}_p$). At each event, one coordinate $p=(\text{counter}\bmod P)$ is restored by the rank-one amount $a=\omega_p-\omega^{\mathrm{eff}}_p$:

```math
\mathbf M\mathrel{+}=a\,\mathbf e_p\mathbf e_p^\top,\quad
\mathbf C_{:,p}\mathrel{+}=a\,\boldsymbol\Theta_{0,:,p},\quad
\tilde{\boldsymbol\Theta}\leftarrow\tilde{\boldsymbol\Theta}+a\bigl(\boldsymbol\Theta_{0,:,p}-\tilde{\boldsymbol\Theta}_{:,p}\bigr)\bigl(\mathbf M'^{-1}\mathbf e_p\bigr)^\top,\quad
\omega^{\mathrm{eff}}_p\leftarrow\omega_p.
```

The last update of $\tilde{\boldsymbol\Theta}$ combines Lemma 8 with the change in $\mathbf C$. It shrinks column $p$ toward the prior.

**Proposition 9 (no wind-up).** Let $W_P(t)$ be the time spanned by the last $P$ replenishments. Then $\mathbf M_t\succeq\min_p\omega_p\,e^{-\kappa_{\max}W_P(t)}\,\mathbf I$. A heartbeat replenishment every $T_{\mathrm{hb}}$ seconds without events guarantees $W_P\le P\,T_{\mathrm{hb}}$. Hence $\lVert\mathbf M^{-1}\rVert_2$ and $\lVert\mathbf U\rVert_2$ are bounded uniformly in time.

**Long-gap rule (spec 0.2.0).** An interval longer than $T_{\mathrm{hb}}$, which can only happen when heartbeats are missing, ends with a *full* restore: every coordinate is replenished at its end time. This makes the bound of Proposition 9 unconditional, $W_P\le P\,T_{\mathrm{hb}}$ for every input stream, because any $P$ consecutive replenishments either all follow intervals no longer than $T_{\mathrm{hb}}$ or include a full restore. The rule is deterministic and decided at the end of the interval, so the batch definition below and Theorem 10 hold with it. It also has a numerical purpose. After a gap of hours without heartbeats the forgetting factor is $d=e^{-\kappa\Delta}\approx e^{-115}$ for $\Delta=10^4$ s, and the square-root update would have to inflate $\mathbf U$ by $d^{-1/2}$ and then apply rank-one updates with $\sigma^2\sim d^{-1}$, losing about 25 digits to cancellation in Lemma 8. The square-root back-end therefore rebuilds $\mathbf U$ from the stored $\mathbf M$ by Cholesky factorization after a full restore (Section 5.4.1).

*Proof.* The data part of $\mathbf M$ is positive semi-definite. Coordinate $p$'s ridge was last restored at most $W_P$ seconds ago and has since decayed by at most $e^{-\kappa_{\max}W_P}$. $\square$

**Scale-free prior strength.** Set $\omega_p=T_\omega\cdot\widehat{\mathbb E}[z_p^2]$, so that the prior weighs as much as $T_\omega$ seconds of typical data (default $T_\omega=1$ s). For a channel that never fires, its column of $\tilde{\boldsymbol\Theta}$ stays at the prior $\boldsymbol\Theta_0$, which is the correct fallback with no wind-up.

**Definition (batch reference).** With $\tau_p(t)$ the time of the last replenishment of coordinate $p$, including full restores (0 initially, that is, the time of the first event):

```math
\mathbf M^{\mathrm{batch}}_t=\int_0^t e^{-\mathcal K(s,t)}\mathbf z(s)\mathbf z(s)^\top ds+\operatorname{diag}\bigl(\omega_p e^{-\mathcal K(\tau_p(t),t)}\bigr),\qquad
\mathbf C^{\mathrm{batch}}_t=\sum_{t_k\le t}e^{-\mathcal K(t_k,t)}\mathbf e_{d_k}\mathbf z(t_k-)^\top+\boldsymbol\Theta_0\operatorname{diag}\bigl(\omega_pe^{-\mathcal K(\tau_p(t),t)}\bigr).
```

**Theorem 10 (the online estimate equals the batch estimate).** In exact arithmetic, after every event $k$, $\tilde{\boldsymbol\Theta}^{\mathrm{online}}(t_k)=\mathbf C^{\mathrm{batch}}_{t_k}(\mathbf M^{\mathrm{batch}}_{t_k})^{-1}$ and $\mathbf U\mathbf U^\top=(\mathbf M^{\mathrm{batch}}_{t_k})^{-1}$. This holds for any piecewise-constant predictable forgetting path $\kappa(\cdot)$, any marks, and any ties.

*Proof.* By induction on the steps of Section 9. The base case is $\mathbf M_0=\operatorname{diag}(\boldsymbol\omega)$, $\mathbf C_0=\boldsymbol\Theta_0\operatorname{diag}(\boldsymbol\omega)$ and $\tilde{\boldsymbol\Theta}_0=\boldsymbol\Theta_0$. Each step updates $(\mathbf M,\mathbf C)$ exactly as the definition changes over that step: Theorem 7 for intervals, the event increment, the ridge decay and restore, and forgetting scaling. Each step also updates $(\mathbf U,\tilde{\boldsymbol\Theta})$ so as to preserve $\mathbf U\mathbf U^\top=\mathbf M^{-1}$ and $\tilde{\boldsymbol\Theta}=\mathbf C\mathbf M^{-1}$, by Lemma 8. Ties change $\mathbf C$ only, since $\Delta=0$ intervals contribute nothing. $\square$

**Measured.** The online and batch estimates agree to 6.8e-14 relative, at 14 checkpoints over 260 events, with two tie batches, an Erlang-2 basis, random marks and a mid-stream switch of $\kappa$. The batch side rebuilds $\mathbf z(s)$ by direct $O(N)$ sums and integrates by adaptive quadrature **[Q5]**.

**Drift control without breaking the $O(1)$ bound.** $\mathbf C$ is stored. At event $k$, one row $r=k\bmod D$ is refreshed exactly as $\tilde\theta_r\leftarrow\mathbf C_{r,:}\mathbf U\mathbf U^\top$, at cost $O(P^2)$. This removes accumulated rounding from incremental updates of $\tilde{\boldsymbol\Theta}$ with a refresh period of $D$ events, at constant worst-case cost.

### 5.4.1 Implementation forms (spec 0.2.0)

Theorem 10 is a statement about $(\mathbf M,\mathbf C)$. Two back-ends maintain them and produce $\tilde{\boldsymbol\Theta}=\mathbf C\mathbf M^{-1}$; both are implemented and verified against the same batch reference.

* **Square-root covariance form** (Lemma 8, the choice of spec 0.1.0). $\mathbf U$ and $\tilde{\boldsymbol\Theta}$ are updated incrementally by Potter updates, at $O((M+1)P^2)$ per event, with the row refresh above. $\mathbf M$ is also stored, for the long-gap refactorization.
* **Information form** (added in 0.2.0). Only $(\mathbf M,\mathbf C,\boldsymbol\omega^{\mathrm{eff}})$ are stored. Each interval adds $\mathbf V^\top\mathbf F\mathbf V$ to $\mathbf M$ directly, using the sparsity of $\mathbf V$ (each coordinate has at most $r_{\max}$ coefficients), and each event solves $\tilde{\boldsymbol\Theta}^\top=\mathbf M^{-1}\mathbf C^\top$ by a fresh Cholesky factorization. The cost is $O(P^3/6+DP^2+(M+1)Pr_{\max})$ per event, which is fewer operations than the square-root form whenever $P/6+D<2.5(M+1)$, which holds for $D\le10$ with the default bank. No incremental state is carried in $\tilde{\boldsymbol\Theta}$, so there is nothing to drift, and the positive-definiteness of $\mathbf M$ is structural: it is a sum of Gram matrices and a positive ridge, scaled by positive factors.

Theorem 11's bound becomes $O(\max(r_{\max}Q^3D^2,\,Q^3D^3))$ for the information form. It is still independent of $N$. The Phase 3 measurements (docs/benchmarks.md) favor the information form at $D=4$ in both speed (2.7 times faster per event) and accuracy (Section 10), and it is the engine's default.

### 5.5 Regime adaptation: time-rescaling CUSUM with variable forgetting

**Time-rescaling theorem** (Brown et al., 2002). Under the correct model, the compensator increments between consecutive events of channel $i$, $\xi_{i,n}=\Lambda_i(t_{i,n})-\Lambda_i(t_{i,n-1})$, are i.i.d. Exp(1). The compensators are accumulated exactly with the served parameters, $\boldsymbol\Lambda\mathrel{+}=\hat{\boldsymbol\Theta}\,\mathbf Z(t,\Delta)$, using Theorem 3, at cost $O(DP)$ per event.

**Detector.** For each channel there are two CUSUMs, one for "events arriving faster than predicted" and one for slower. Each uses the log-likelihood ratio of Exp($\varrho$) against Exp(1):

```math
C^{\uparrow}_i\leftarrow\max\bigl(0,\,C^{\uparrow}_i+\ln\varrho-(\varrho-1)\,\xi\bigr),\qquad
C^{\downarrow}_i\leftarrow\max\bigl(0,\,C^{\downarrow}_i-\ln\varrho-(\varrho^{-1}-1)\,\xi\bigr),\qquad\varrho>1.
```

When $\max_i\max(C^\uparrow_i,C^\downarrow_i)>b_{\mathrm{cs}}$, the next intervals use $\kappa_{\mathrm{fast}}$ for a dwell time $\tau_{\mathrm{dwell}}$, and all CUSUMs are reset (Page, 1954).

* **Exactness is preserved.** $\kappa$ is decided at an event and applies to the next interval, so it is predictable. Theorem 10 covers any such path, and [Q5] includes a switch.
* **Batch rule (spec 0.2.0).** Alarms are detected per event, but the reset and the start of the dwell take effect when the tie batch closes, at the first strictly later timestamp. Because $\kappa$ is only used for later intervals, this changes nothing for distinct timestamps, and it makes the detector invariant to the order of events inside a batch, as Proposition 4 requires.
* **Graceful behavior.** During fast forgetting the data weight shrinks while the ridge is replenished, so estimates shrink toward the static calibration $\boldsymbol\Theta_0$ and are then pulled toward the new regime.
* **Calibration.** The false-alarm rate falls roughly like $e^{-b_{\mathrm{cs}}}$. $b_{\mathrm{cs}}$ is set by simulating the target in-control average run length (Phase 3/5). The residuals use plug-in parameters, so they are only approximately Exp(1).

### 5.6 Complexity

**Theorem 11 (per-event complexity).** Let $r_{\max}=\max_q r_q$, so $M+1\le1+r_{\max}Q$, and $P=1+DQ$. Let the projection use a fixed number $I_b$ of bisection steps (Section 6.4), and let the Neumann series use $K$ terms (Section 7). The per-event time is

```math
\mathcal T=O\bigl((M+1)(P^2+DP)\bigr)+O(P^2)+O(DQ\,r_{\max}^2)+O(I_b\,D^2Q)+O(K D^2)
=O\bigl(r_{\max}\,Q^3D^2\bigr)\quad(\text{fixed }I_b,K),
```

the memory is $O(P^2+DP+DQr_{\max})=O(Q^2D^2)$, and exactly $Q+1$ exponentials are evaluated per event. None of these depend on $N$ or on the event history. For a fixed kernel bank the cost is $O(D^2)$, which is optimal up to constants for any method that touches all $D\cdot P=\Theta(D^2Q)$ parameters, or evaluates all $D$ intensities, at every event.

*Proof.* The engine state is the fixed-size tuple $(\mathbf U,\tilde{\boldsymbol\Theta},\mathbf C,\boldsymbol\omega^{\mathrm{eff}},R,\boldsymbol\pi,\hat{\boldsymbol\Theta},\mathbf v,\boldsymbol\lambda^\infty,\boldsymbol\Lambda,C^{\uparrow\downarrow},\text{mark statistics},\text{counters})$. Each event maps the state to a new state of the same size by the finite sequence of Section 9. The cost of each step is:

* flush $O(DQ)$;
* $\mathbf F$ and pivoted Cholesky $O((M+1)^3)$;
* forming $\mathbf w_r$: $O((M+1)P)$;
* at most $M+1$ Potter updates, $O(P^2)$ each, with $\tilde{\boldsymbol\Theta}$ updates $O(DP)$ each;
* state decay $O(DQr_{\max}^2)$;
* event increment and replenishment $O(P^2)$;
* row refresh $O(P^2)$;
* intensities and compensators $O(DP)$;
* projection $O(I_bD\cdot DQ)$;
* Perron step and Jacobi sweep $O(D^2+D^2Q)$;
* forecasts $O(DP)$;
* Neumann series $O(KD^2)$.

By induction on $k$, the state size and the work per event are bounded by constants independent of $k$. $\square$

**Size of the default configuration** ($D=4$, $Q=4$ with two Erlang-2 bases, so $P=17$ and $M+1=7$): about $7\times3\times17^2\approx6{,}100$ flops of Potter updates, $\approx1{,}000$ for $\tilde{\boldsymbol\Theta}$, $\approx600$ for $\mathbf F$, $\approx1{,}200$ for the event and replenishment updates, and under 1,000 for everything else. That is roughly $10^4$ flops in total, of which inference (intensities, toxicity, quotes) is under 500. The real latency is to be *measured* in Phase 3; no number is claimed here.

If the full update cannot meet the sub-microsecond target, there are three fallbacks:

1. A smaller bank. With $Q=3$, all Erlang-1, the count is about $3\times10^3$ flops.
2. The exploitation of the rank adaptivity shown in [Q4].
3. Running the estimator on a second core from the same event stream and publishing $\hat{\boldsymbol\Theta}$ through a seqlock, so that the hot thread performs only $O(D^2Q)$ inference.

Every option keeps $O(1)$ per event and zero allocation.

### 5.7 Statistical properties

**Proposition 12.**

(a) *Variance.* For a stationary regime and $\kappa\to0$, $\operatorname{Cov}(\tilde\theta_i)\approx\frac{\kappa}{2}\,\boldsymbol\Sigma^{-1}\boldsymbol\Gamma_i\boldsymbol\Sigma^{-1}$, with $\boldsymbol\Sigma=\mathbb E[\mathbf z\mathbf z^\top]$ and $\boldsymbol\Gamma_i=\mathbb E[\mathbf z\mathbf z^\top\lambda_i]$.

(b) *Lag.* Suppose the true parameters step from $\theta_1$ to $\theta_2$ at time $s_0$. Replacing $\mathbf M,\mathbf C$ by their expectations, $\bar\theta(s_0+\tau)=(\mathbf W_1+\mathbf W_2)^{-1}(\mathbf W_1\theta_1+\mathbf W_2\theta_2)$, with $\mathbf W_1=e^{-\kappa\tau}\boldsymbol\Sigma_1$ and $\mathbf W_2=(1-e^{-\kappa\tau})\boldsymbol\Sigma_2$. If $\boldsymbol\Sigma_1=\boldsymbol\Sigma_2$, the estimate covers half the step after exactly one half-life $\ln2/\kappa$.

(c) *Efficiency.* $\boldsymbol\Sigma^{-1}\boldsymbol\Gamma_i\boldsymbol\Sigma^{-1}\succeq\bigl(\mathbb E[\mathbf z\mathbf z^\top/\lambda_i]\bigr)^{-1}$, which is the inverse Fisher information per unit time, with equality if and only if $\lambda_i$ is almost surely constant. Least squares therefore loses efficiency in proportion to how bursty the intensity is.

*Proof sketch.* (a) $\mathbf c-\mathbf M\theta^\star=\int w\,\mathbf z\,dM_i$ has covariance $\int w^2\,\mathbb E[\mathbf z\mathbf z^\top\lambda_i]\approx\boldsymbol\Gamma_i/(2\kappa)$, while $\mathbf M\approx\boldsymbol\Sigma/\kappa$. (b) $\mathbb E\int w\,\mathbf z\,dN_i=\mathbb E\int w\,\mathbf z\mathbf z^\top\theta$ on each regime, split at $s_0$. (c) This is the Godambe–Heyde optimality of the score among unbiased linear estimating equations, which follows from Cauchy–Schwarz applied to $\mathbf z/\sqrt{\lambda}$ and $\mathbf z\sqrt\lambda$. $\square$

**Choosing the forgetting rate.** Variance grows like $\kappa$ and lag shrinks like $1/\kappa$. For parameters drifting as a random walk with diffusion $q$, the steady-state MSE is $\approx q/(2\kappa)+\kappa c/2$, minimized at $\kappa^\star=\sqrt{q/c}$. The CUSUM of Section 5.5 removes the need to commit to one value: slow forgetting in calm regimes, fast forgetting after a detected break.

### 5.8 Validation experiment (prototype, not the production benchmark)

This is a Python design-validation run (`--full` mode of the script). It is not a benchmark of the engine.

**Setup.** $D=2$, one exponential kernel ($\beta=40$/s), $\mu=(12,8)$/s, simulated by Ogata thinning with parameters at evaluation time. The cross-venue coupling $\alpha_{1\leftarrow0}$ steps from $0.10$ to $0.55$ at $t=600$ s, with $T=1200$ s, about 43,000 events per run, and 8 seeds. Calm-period metrics are measured on $t\in[300,599]$ s. "Lag" is the time after the step until the estimate first crosses the midpoint 0.325.

**[F1] Estimator comparison.**

| Estimator | Half-life | Bias | SD | RMSE (calm) | Lag to 50% of step |
|---|---|---|---|---|---|
| online EM | 3 s | +0.0675 | 0.0289 | 0.0734 | 5.1 s |
| online EM | 6 s | +0.0385 | 0.0187 | 0.0428 | 14.3 s |
| online EM | 12 s | +0.0227 | 0.0124 | 0.0259 | 39.2 s |
| online EM | 24 s | +0.0220 | 0.0102 | 0.0243 | 86.8 s |
| **discounted LS** | 6 s | +0.0016 | 0.0423 | 0.0424 | 5.8 s |
| **discounted LS** | 12 s | +0.0010 | 0.0301 | 0.0302 | 11.9 s |
| **discounted LS** | 24 s | +0.0008 | 0.0219 | 0.0219 | 22.1 s |
| **discounted LS** | 48 s | +0.0009 | 0.0155 | 0.0155 | 49.0 s |
| recursive Newton ML | 6 s | −0.0022 | 0.0390 | 0.0391 | 10.8 s |
| recursive Newton ML | 12 s | −0.0019 | 0.0284 | 0.0284 | 22.2 s |
| recursive Newton ML | 24 s | −0.0007 | 0.0201 | 0.0201 | 47.0 s |
| recursive Newton ML | 48 s | +0.0010 | 0.0140 | 0.0140 | 97.4 s |

At matched lag, least squares has the lowest error. At about 6 s lag its RMSE is 0.042, against 0.073 for EM. At about 12 s it is 0.030, against 0.039 for Newton and 0.043 for EM. At about 22 s it is 0.022, against 0.028 for Newton. Its lag equals the half-life to within 8%, as Proposition 12(b) predicts.

**[F2] Adaptive forgetting** (two-sided CUSUM, $\varrho=1.5$).

| Slow half-life | Fast half-life | Dwell | Threshold $b_{\mathrm{cs}}$ | RMSE (calm) | Lag to 50% | Alarms per run |
|---|---|---|---|---|---|---|
| 24 s | – | – | – | 0.0219 | 22.1 s | 0.0 |
| 24 s | 3 s | 6 s | 14 | 0.0219 | 16.5 s | 1.0 |
| 48 s | – | – | – | 0.0155 | 49.0 s | 0.0 |
| 48 s | 4 s | 8 s | 14 | 0.0155 | 22.6 s | 1.8 |

**Limitations.** This is one well-specified scenario with $D=2$ and a single kernel, run in a Python prototype. Proposition 12(c) predicts that likelihood methods gain ground when intensities are very bursty. The full comparison belongs in Phase 3 (synthetic, $D=4$, the full bank) and Phase 5 (real data).

---

## 6. Stability and the sub-criticality guardrail

### 6.1 The certificate

**Lemma 13 (Collatz–Wielandt bound).** For $\mathbf A\ge0$ and any $\mathbf v>0$, $\rho(\mathbf A)\le\max_i(\mathbf A\mathbf v)_i/v_i$. Equality holds when $\mathbf v$ is a Perron vector of an irreducible $\mathbf A$.

*Proof.* Let $r=\max_i(\mathbf A\mathbf v)_i/v_i$. Then $\mathbf A\mathbf v\le r\mathbf v$, and by nonnegativity $\mathbf A^n\mathbf v\le r^n\mathbf v$. In the weighted max-norm $\lVert\mathbf x\rVert_{\mathbf v}=\max_i\lvert x_i\rvert/v_i$, the induced norm of a nonnegative matrix is $\max_i(\mathbf A\mathbf v)_i/v_i$, so $\lVert\mathbf A^n\rVert_{\mathbf v}\le r^n$ and $\rho(\mathbf A)=\lim_n\lVert\mathbf A^n\rVert_{\mathbf v}^{1/n}\le r$ (Horn and Johnson, 2013, Ch. 8). $\square$

Verified on 3,000 random nonnegative matrices, where $\rho-\text{bound}\le1.1\times10^{-16}$ (rounding level) **[Q7]**.

**The sub-critical set is not convex, but an inner polytope is.** The set $\lbrace\mathbf A\ge0:\rho(\mathbf A)\le\rho_{\max}\rbrace$ is not convex in $\mathbf A$. For example, the matrices with a single off-diagonal entry 2 in position (1,2) or (2,1) both have $\rho=0$, but their midpoint has $\rho=1$. For a fixed $\mathbf v>0$, however,

```math
\mathcal P(\mathbf v)=\Bigl\lbrace\boldsymbol\alpha\ge0:\ \sum_{j,q}\alpha^q_{ij}v_j\le\rho_{\max}v_i\ \ \forall i\Bigr\rbrace
```

is a convex polytope whose constraints separate by row, and every point of it has $\rho(\mathbf G)\le\rho_{\max}$ by Lemma 13.

### 6.2 Perron-vector tracking

Once per event, $\mathbf v\leftarrow\Pi_{\mathcal B}\bigl((\hat{\mathbf G}\mathbf v+\epsilon_v\mathbf v_{\mathrm{ref}})/\lVert\cdot\rVert_\infty\bigr)$, where $\mathcal B=\lbrace\mathbf v:\chi^{-1/2}\mathbf v_{\mathrm{ref}}\le\mathbf v\le\chi^{1/2}\mathbf v_{\mathrm{ref}}\rbrace$, $\chi\ge1$, and $\mathbf v_{\mathrm{ref}}$ is the Perron vector of $\boldsymbol\Theta_0$'s branching matrix (or $\mathbf 1$). This costs $O(D^2)$. The $\epsilon_v$ term keeps $\mathbf v>0$ even when $\hat{\mathbf G}$ is reducible. The box makes $\mathbf v$ ratios bounded across time, which Theorem 15 needs. Warm-started power iteration tightens the certificate as parameters move slowly.

### 6.3 The criticality index

Let $\tilde{\mathbf G}$ be the branching matrix of $\tilde{\boldsymbol\Theta}$ with negative entries clamped to zero, and define $\tilde\rho(t)=\max_i(\tilde{\mathbf G}\mathbf v)_i/v_i$. This is an upper bound on how critical *the data* currently look, before projection. A value near or above 1 means an endogenous cascade: the reflexivity indicator of Filimonov and Sornette (2012) and Hardiman, Bercot and Bouchaud (2013), here available every event. It feeds the global kill switch (Section 8.6).

### 6.4 Serving projection

The served parameters $\hat{\boldsymbol\Theta}$ are obtained from $\tilde{\boldsymbol\Theta}$ in two steps.

* Baselines: $\hat\mu_i=\operatorname{clip}(\tilde\mu_i,\mu_{\min},\mu_{\max})$.
* Each excitation row: the Euclidean projection of $\tilde{\mathbf a}_i=(\tilde\alpha^q_{ij})_{j,q}$ onto $\lbrace\mathbf a\ge0,\ \mathbf w^\top\mathbf a\le\rho_{\max}v_i\rbrace$, with $w_{jq}=v_j$.

**Proposition 14 (exact $O(DQ)$ row projection).** The projection is $\mathbf a^\star=(\tilde{\mathbf a}-\nu^\star\mathbf w)_+$. If $\mathbf w^\top(\tilde{\mathbf a})_+\le\rho_{\max}v_i$, then $\nu^\star=0$. Otherwise $\nu^\star>0$ is the unique root of the continuous, strictly decreasing, piecewise-linear function $\phi(\nu)=\mathbf w^\top(\tilde{\mathbf a}-\nu\mathbf w)_+-\rho_{\max}v_i$.

The hot path computes it with a deterministic, iteration-free cost of $O(n^2)$ for $n=DQ$ (spec 0.2.0; 0.1.0 used 64 bisection steps, $O(64n)$, which is more work for $n\le64$ and needs a tolerance):

1. Evaluate $\phi$ at every breakpoint $\nu_m=\tilde a_m/w_m$ with $\tilde a_m>0$. Let $\nu_L$ be the largest breakpoint with $\phi(\nu_L)>0$, or 0. The active set is $\mathcal A=\lbrace m:\nu_m>\nu_L\rbrace$.
2. Finalize exactly: $\nu^\star=\bigl(\sum_{\mathcal A}w_m\tilde a_m-\rho_{\max}v_i\bigr)/\sum_{\mathcal A}w_m^2$.
3. Apply a final feasibility rescale if rounding leaves $\mathbf w^\top\mathbf a^\star>\rho_{\max}v_i$.

The test reference is the exact $O(n\log n)$ sorting algorithm.

*Proof.* These are the KKT conditions of the strictly convex QP $\min\lVert\mathbf a-\tilde{\mathbf a}\rVert^2$ subject to $\mathbf a\ge0$ and $\mathbf w^\top\mathbf a\le c$. Monotonicity of $\phi$ gives a unique root. On the true active set $\phi$ is affine, so the finalization is exact. $\square$

**Measured.** The largest KKT residual over 3,000 random rows was 1.6e-15 **[Q7]**.

**Corollary.** $\rho(\hat{\mathbf G}(t))\le\rho_{\max}$ at every event, certified, in $O(D^2Q)$ time. Scaling all of $\alpha$ by a common factor was rejected as an alternative. It is feasible but not a projection, and it distorts the contagion structure. The row projection shrinks most the parents that matter most systemically, which are those with large $v_j$.

### 6.5 Non-explosion of the served model

**Theorem 15.** Assume:

* (S1) for every $t$, the served parameters satisfy $\hat\alpha\ge0$, $\hat\mu\le\mu_{\max}$ and $\hat{\mathbf G}(t)\mathbf v(t)\le\rho_{\max}\mathbf v(t)$;
* (S2) $\mathbf v(t)\in\mathcal B$, so that $v_j(s)/v_j(t)\le\chi$ for all $s,t$;
* (S3) $\mathbb E[g_k\mid\mathcal F_{t_k-}]\le\bar g$;
* (S4) $\bar g\,\chi\,\rho_{\max}<1$.

Then for the point process generated by the served model with the given parameter path, the mean intensity satisfies

```math
\sup_{t\ge0}\ \max_i\ \mathbb E\,\lambda_i(t)\ \le\ \frac{\chi^{1/2}\,\max_i v_{\mathrm{ref},i}}{\chi^{-1/2}\min_i v_{\mathrm{ref},i}}\cdot\frac{\mu_{\max}}{1-\bar g\chi\rho_{\max}}.
```

*Proof.* Let $m_j(s)=\mathbb E\lambda_j(s)$ and $\mathcal M(t)=\sup_{s\le t}\max_j m_j(s)/v_j(s)$. $\mathcal M(t)$ is finite for each $t$ by Gronwall's inequality, since kernels are bounded on finite horizons. By (S3) and $\int_0^\infty h_q=1$,

```math
m_i(t)\le\mu_{\max}+\bar g\sum_{j,q}\hat\alpha^q_{ij}(t)\int_0^t h_q(t-s)\,m_j(s)\,ds
\le\mu_{\max}+\bar g\sum_j\hat G_{ij}(t)\,\chi\,v_j(t)\,\mathcal M(t)
\le\mu_{\max}+\bar g\chi\rho_{\max}\,v_i(t)\,\mathcal M(t),
```

using $m_j(s)\le\mathcal M(t)v_j(s)\le\mathcal M(t)\chi v_j(t)$ and then (S1). Divide by $v_i(t)\ge\chi^{-1/2}\min v_{\mathrm{ref}}$ and take the supremum over $s\le t$: $\mathcal M(t)\le\frac{\mu_{\max}}{\chi^{-1/2}\min v_{\mathrm{ref}}}+\bar g\chi\rho_{\max}\mathcal M(t)$. Solve for $\mathcal M(t)$ using (S4), then multiply by $\max v\le\chi^{1/2}\max v_{\mathrm{ref}}$. $\square$

**Interpretation.**

* *What is bounded.* Forecasts, stationary baselines and anything simulated from the served model stay uniformly bounded in time, even when the data are in a liquidation cascade. The intensities driven by the observed data are always finite (Section 4.6). What the guardrail prevents is the model *extrapolating* an explosion.
* *The cost of the guarantee.* It holds under (S3), and normalized marks give $\bar g\approx1$ (Section 2.3). A fully robust guarantee with $\bar g=g_{\max}$ would need $\rho_{\max}<1/(\chi g_{\max})$, which is much too conservative for crypto order flow with branching ratios near 0.9. The robust mode exists as a configuration flag; the default is the normalized-mark mode.

### 6.6 Reaction speeds

| Mechanism | Reacts within | Section |
|---|---|---|
| Intensities, forecasts, toxicity, quotes | the same event (state jump) | 4, 7, 8 |
| Cross-venue lead-lag | the same event: a venue-A sweep raises the venue-B forecast before venue B trades | 7.3 |
| Criticality index | the same event | 6.3 |
| Parameters, calm regime | the slow half-life (default 60 s) | 5.5 |
| Parameters after a detected break | detection delay plus the fast half-life (default 5 s) | 5.5 |

The millisecond reaction asked for in the research question comes from the *state* and the *forecasts*. Parameter adaptation keeps that state correctly calibrated across regimes; it does not, and cannot statistically, move in milliseconds.

---

## 7. Directional cross-venue toxicity

### 7.1 Exact forecast from the current state

For horizon $H$, the expected number of channel-$i$ arrivals in $(t,t+H]$ from the baseline and from already-observed parents, excluding descendants of future arrivals, is

```math
m^{(0)}_i(t;H)=\hat\mu_iH+\sum_{j,q}\hat\alpha^q_{ij}\,X^q_j(t+,H),
```

where $X^q_j$ is Theorem 3 evaluated on the post-event state $R+\boldsymbol\pi$. Because $H$ is fixed, $\psi_n(\beta_qH)$ and $e^{-\beta_qH}$ are precomputed constants, so the forecast needs no exponential on the hot path and costs $O(DP)$.

**Proposition 16 (bracketing the full forecast).** Let $\mathbf m(t;H)$ be the full expected count, including descendants of arrivals inside the window. Then $\mathbf m^{(0)}\le\mathbf m\le(\mathbf I-\hat{\mathbf G})^{-1}\mathbf m^{(0)}$.

*Proof.* $\mathbf m=\mathbf m^{(0)}+\int_0^H\boldsymbol\Phi(H-s)\,d\mathbf m(s)$, where $\Phi_{ij}(u)=\sum_q\hat\alpha^q_{ij}\int_0^uh_q\le\hat G_{ij}$. Hence $\mathbf m\ge\mathbf m^{(0)}$, and $\mathbf m\le\mathbf m^{(0)}+\hat{\mathbf G}\mathbf m$. Multiplying by $(\mathbf I-\hat{\mathbf G})^{-1}=\sum_n\hat{\mathbf G}^n\ge0$, which converges since $\rho(\hat{\mathbf G})<1$, preserves the inequality. $\square$

The $K$-term Neumann approximation $\sum_{n\le K}\hat{\mathbf G}^n\mathbf y$ has error at most $\rho_{\max}^{K+1}\lVert\mathbf y\rVert_{\mathbf v}/(1-\rho_{\max})$ in the $\mathbf v$-norm (Lemma 13). The default is $K=3$.

### 7.2 Stationary reference

$\boldsymbol\lambda^\infty=(\mathbf I-\hat{\mathbf G})^{-1}\hat{\boldsymbol\mu}$ is tracked with one warm-started Jacobi sweep per event, $\boldsymbol\lambda^\infty\leftarrow\hat{\boldsymbol\mu}+\hat{\mathbf G}\boldsymbol\lambda^\infty$, at cost $O(D^2)$. This is a $\rho_{\max}$-contraction in the $\mathbf v$-norm, so its tracking error obeys $e_k\le\rho_{\max}e_{k-1}+\delta_k$, where $\delta_k$ is the per-event movement of the fixed point. Parameters move slowly, so $\delta_k$ is tiny.

**Campbell means.** For any stationary process with mean intensity $\lambda^\infty_j$ and unit-mean marks, $\mathbb E\,R^{q,(n)}_j=\lambda^\infty_j\int_0^\infty\frac{s^n}{n!}e^{-\beta_qs}ds=\lambda^\infty_j/\beta_q^{\,n+1}$. This needs only the first moment, not a Poisson assumption. The Monte Carlo check on a stationary Hawkes process with an Erlang-2 kernel matched to 0.25% **[Q8]**.

Hence the stationary forecast is

```math
\bar m^{(0)}_i(H)=\hat\mu_iH+\sum_{j,q}\hat\alpha^q_{ij}\,\lambda^\infty_j\,\bar k_q(H),\qquad
\bar k_q(H)=\sum_{l=0}^{r_q-1}\beta_q^{\,r_q-1-l}\,\frac{H^{\,r_q-l}}{(r_q-1-l)!}\,\psi_{r_q-1-l}(\beta_qH).
```

In particular $\bar k=H\psi_0(\beta H)$ for Erlang-1 and $\bar k=H\psi_0(\beta H)+\beta H^2\psi_1(\beta H)$ for Erlang-2. These follow by substituting the Campbell means into Theorem 3. Equivalently $\bar k_q(H)=\int_0^H S_q(u)\,du$ with $S_q$ the kernel's survival function, which tends to the kernel mean $r_q/\beta_q$ as $H\to\infty$. (Spec 0.1.0 printed the power of $\beta_q$ in the general sum as $l$; the special cases were correct.)

### 7.3 The toxicity score

Each channel has a direction $s_i$ and an impact weight $\eta_i\ge0$. $\eta_i$ is the expected absolute mid-price move at the quoting venue per channel-$i$ event, calibrated offline by cross-impact regression in Phase 5. Define the impact-weighted forecast flows and their excess over stationarity:

```math
\mathcal M^\pm=\sum_{i:\,s_i=\pm1}\eta_i\,m^{(0)}_i,\qquad
\bar{\mathcal M}^\pm=\sum_{i:\,s_i=\pm1}\eta_i\,\bar m^{(0)}_i,\qquad
E^\pm=\bigl(\mathcal M^\pm-\bar{\mathcal M}^\pm\bigr)_+ .
```

```math
\boxed{\;\tau^\pm(t)=\underbrace{\frac{E^\pm}{\mathcal M^\pm}}_{\text{anomalous fraction}}\cdot\underbrace{\frac{(E^\pm-E^\mp)_+}{E^++E^-}}_{\text{directional purity}}\;}
```

**Proposition 17.**

1. $\tau^\pm\in[0,1)$.
2. *Exclusivity:* $\tau^+\tau^-=0$ at all times.
3. *Two-sided invariance:* if $E^+=E^-$, then $\tau^\pm=0$ however large the volume. Benign two-sided bursts score zero.
4. *Unit invariance:* $\tau$ is unchanged by rescaling $\eta$, or by rescaling forecasts and baselines together.
5. *Monotonicity:* when $E^+>E^-$, $\tau^+$ is non-decreasing in $\mathcal M^+$.
6. *Lead-lag:* a venue-A buy event raises $m^{(0)}_{B+}$ immediately through $\hat\alpha^q_{B+,A+}X^q$, even for an Erlang-2 basis whose intensity is still zero at $t+$, because $X^q(t+,H)$ includes the hump mass $\int_0^H h_q$ that is still to come.

*Proof.* (1) The anomalous fraction is $1-\bar{\mathcal M}^\pm/\mathcal M^\pm<1$ when positive (since $\bar{\mathcal M}>0$, because $\hat\mu\ge\mu_{\min}>0$), and the purity lies in $[0,1]$. (2) At most one of $(E^+-E^-)_+$ and $(E^--E^+)_+$ is nonzero. (3) Immediate. (4) Both factors are homogeneous of degree 0. (5) Both factors are non-decreasing in $\mathcal M^+$ in that region. (6) By Theorem 3, $X^q\ge\beta^r R^{(0)}\Delta^r\psi_{r-1}/(r-1)!>0$ after the jump of $R^{(0)}$. $\square$

**Economic reading.** $\tau^+$ estimates the probability that buy-side flow arriving over the next $H$ is *anomalous and one-sided*. That is the operational definition of informed flow used in Section 8. Two diagnostics are reported alongside it.

* The cross-venue attribution share: the fraction of $E^+$ whose parents are on the other venue, computed from the per-$(j,q)$ terms.
* The criticality index $\tilde\rho$.

**Horizon.** $H=\mathcal L_{\mathrm{rt}}+H_d$, where $\mathcal L_{\mathrm{rt}}$ is the cancel round-trip latency. A resting quote cannot be withdrawn sooner than $\mathcal L_{\mathrm{rt}}$, so the forecast must cover at least that window. The default is $H=50$ ms.

### 7.4 Total excess cascade

For the reservation price we need the *total* expected excess arrivals of the current cascade, $\mathbf X^\infty=\lim_{H\to\infty}\bigl(\mathbf m(t;H)-H\boldsymbol\lambda^\infty\bigr)$.

**Proposition 18.**

```math
\mathbf X^\infty=(\mathbf I-\hat{\mathbf G})^{-1}\mathbf y^\infty,\qquad
y^\infty_i=\sum_{j,q}\hat\alpha^q_{ij}\Bigl(\sum_{l=0}^{r_q-1}\beta_q^{\,l}R^{q,(l)}_j(t+)\ -\ \lambda^\infty_j\,\frac{r_q}{\beta_q}\Bigr).
```

In particular, the remaining kernel mass is $R^{(0)}$ for Erlang-1 and $R^{(0)}+\beta R^{(1)}$ for Erlang-2.

*Proof.* Let $\varepsilon(s)=\mathbb E[\boldsymbol\lambda(t+s)\mid\mathcal F_t]-\boldsymbol\lambda^\infty$. Subtracting the stationary renewal identity from the conditional one gives $\varepsilon(s)=\mathbf c(s)-\int_s^\infty\boldsymbol\phi(v)\,dv\,\boldsymbol\lambda^\infty+\int_0^s\boldsymbol\phi(s-u)\varepsilon(u)\,du$, where $\mathbf c(s)$ is the current-history excitation and $\boldsymbol\phi_{ij}=\sum_q\hat\alpha^q_{ij}h_q$. Integrate over $s\in[0,\infty)$. $\int_0^\infty\mathbf c=$ the remaining mass terms; $\int_0^\infty\int_s^\infty\boldsymbol\phi=\sum_q\hat\alpha^q\cdot\text{mean}(h_q)=\sum_q\hat\alpha^q\,r_q/\beta_q$; and $\int\boldsymbol\phi=\hat{\mathbf G}$. Solve the resulting linear equation. $\square$

$\mathbf y^\infty$ can be negative, for example in a lull, and the reservation skew then reverses. That is correct. It is computed with the $K$-term Neumann series and the error bound of Section 7.1.

---

## 8. Market-making control law

### 8.1 Model

The market maker quotes on venue $V$ (default: spot venue B, with futures venue A as the leading signal). It has inventory $q\in[-q_{\max},q_{\max}]$, CARA risk aversion $\gamma$, and an inventory-risk horizon $T_h$, which plays the role of $T-t$ in Avellaneda and Stoikov's stationary approximation.

* **(A1) Price.** $dS=\sigma_W\,dW+\sum_i s_iJ_i\,dN_i$. Aggressive arrivals move the mid at the quoting venue, including cross-venue arrivals through cross-impact.
* **(A2) Informed flow (Glosten and Milgrom, 1985).** An arrival is informed with conditional probability $\tau^{s_i}(t)$. Its impact is $J_i=J^b_i+\mathbb 1_{\mathrm{inf}}J^I_i$, so the conditional mean impact is $\bar J_i(t)=J^b_i+\tau^{s_i}(t)J^I_i$.
* **(A3) Fills.** An aggressive buy on $V$ fills our ask at distance $\delta^a$ from the mid with probability $e^{-k\delta^a}$ (Avellaneda and Stoikov's exponential fill model, with the arrival rate supplied by the Hawkes intensity). The sell side is symmetric.
* **(A4) No benign trend.** $\sum_is_iJ^b_i\lambda^\infty_i=0$ at HFT horizons.
* **(A5) Small-impact regime.** $\gamma J\ll1$, with the frozen-inventory approximation of Avellaneda and Stoikov.

### 8.2 Quotes

**Theorem 19.** Under (A1)–(A5), to first order in $\gamma J$, the optimal quotes are

```math
p^a=s+\hat\Delta-q\gamma\sigma^2T_h+\tfrac12\gamma\sigma^2T_h+\tfrac1\gamma\ln\!\bigl(1+\tfrac\gamma k\bigr)+\varepsilon^+,\qquad
p^b=s+\hat\Delta-q\gamma\sigma^2T_h-\tfrac12\gamma\sigma^2T_h-\tfrac1\gamma\ln\!\bigl(1+\tfrac\gamma k\bigr)-\varepsilon^-,
```

```math
\varepsilon^\pm=J^b_{V\pm}+\tau^\pm J^I_{V\pm},\qquad
\hat\Delta=\sum_i s_i\,\bar J_i(t)\,X^\infty_i,\qquad
\sigma^2=\sigma_W^2+\frac1H\sum_i m^{(0)}_i(t;H)\,\mathbb E[J_i^2],\qquad
\mathbb E[J_i^2]=(J^b_i)^2+\tau^{s_i}\bigl((J^b_i+J^I_i)^2-(J^b_i)^2\bigr)\ \text{under (A2)}.
```

*Derivation.* Use the ansatz $u(t,x,s,q)=-\exp\bigl(-\gamma(x+qs+\vartheta_q(t))\bigr)$.

* **Arrivals that do not fill us.** Their contribution is $u\sum_i\lambda_i\,\mathbb E[e^{-\gamma qs_iJ_i}-1]\approx u\,\bigl(-\gamma q\nu+\tfrac12\gamma^2q^2\sum_i\lambda_i\mathbb E J_i^2\bigr)$, with drift $\nu=\sum_i\lambda_is_i\bar J_i$. So jumps add $\sum\lambda_i\mathbb EJ_i^2$ to the variance, and their drift enters $\vartheta$.
* **An ask fill.** The arrival moves the mid to $s+J$ whether or not we are filled. Filled, we hold $x+s+\delta^a$ and $q-1$ units; not filled, $x$ and $q$ units, both marked at $s+J$. The wealth difference is $\delta^a-J+\vartheta_{q-1}-\vartheta_q$. The ask term is therefore $\lambda_{V+}\sup_\delta e^{-k\delta}\,u\,e^{-\gamma qJ}\bigl[e^{-\gamma(\delta-J+\vartheta_{q-1}-\vartheta_q)}-1\bigr]$. To first order in $\gamma J$, replace $J$ by its conditional mean $\varepsilon^+$ and $e^{-\gamma qJ}$ by 1. The quantity to maximize is then $e^{-k\delta}\bigl(1-e^{-\gamma(\delta-c)}\bigr)$ with $c=\varepsilon^+-(\vartheta_{q-1}-\vartheta_q)$. Setting its derivative to zero gives $e^{-\gamma(\delta-c)}(k+\gamma)=k$, so $\delta^{a\star}=c+\frac1\gamma\ln(1+\frac\gamma k)$.
* **Frozen-inventory approximation.** This gives $\vartheta_q=q\hat\Delta-\frac12\gamma\sigma^2q^2T_h$, with the transient drift integrated into $\hat\Delta$ (Proposition 18 and (A4)). Then $\vartheta_{q-1}-\vartheta_q=-\hat\Delta+\frac12\gamma\sigma^2T_h(2q-1)$, and $p^a=s+\delta^{a\star}$ is the stated formula.
* **The bid** is symmetric, with $\vartheta_{q+1}-\vartheta_q=\hat\Delta-\frac12\gamma\sigma^2T_h(2q+1)$. $\square$

**How each requirement of the research question is met.**

* **Reservation-price skew against adverse flow.** $\hat\Delta>0$ when the current cascade is buy-heavy. Both quotes then move up: we stop selling cheaply and buy before the move.
* **Asymmetric widening of the endangered side.** Only the ask carries $\tau^+J^I_{V+}$, only the bid carries $\tau^-J^I_{V-}$, and by exclusivity at most one is nonzero at a time.
* **Benign volatility widens both sides.** Two-sided bursts enter through $\sigma^2$, symmetrically, and $\tau=0$ for them.
* **Fill rate.** To this order the quotes do not depend on the fill-intensity level, which is a known property of the Avellaneda–Stoikov approximation. The Hawkes intensity affects the control through $\sigma^2$, $\hat\Delta$ and $\tau$.

### 8.3 When to pull a side

Let $\delta_{\max}$ be the largest useful quoting distance, set by the price band, the queue-priority value, or the message budget. The ask objective $e^{-k\delta}(1-e^{-\gamma(\delta-c^a)})$ is unimodal. It is negative for $\delta<c^a$ and positive beyond, and the break-even distance is

```math
c^a(t)=\varepsilon^+(t)+\hat\Delta(t)-\tfrac12\gamma\sigma^2T_h(2q-1).
```

The constrained optimum is therefore $\min(\delta^{a\star},\delta_{\max})$ when $c^a<\delta_{\max}$. When $c^a\ge\delta_{\max}$, every admissible ask has negative value, and **cancelling is optimal**. Because $c^a$ is affine in $\tau^+$ with slope $J^I_{V+}>0$, this is exactly a toxicity threshold:

```math
\theta^a_{\mathrm{on}}(t)=\frac{\delta_{\max}-\bigl(J^b_{V+}+\hat\Delta-\tfrac12\gamma\sigma^2T_h(2q-1)\bigr)}{J^I_{V+}}\quad\text{(clipped to }[0,1]).
```

The bid is symmetric. The threshold is derived rather than tuned, and it depends on inventory: a long market maker is slower to pull its ask.

### 8.4 Hysteresis and cooldown

Each side runs a two-state Schmitt trigger. It goes from ACTIVE to PULLED when $\tau\ge\theta_{\mathrm{on}}$. It goes from PULLED back to ACTIVE when $\tau\le\theta_{\mathrm{on}}-\Delta_\tau$ **and** at least $c_{\min}$ seconds have passed since the pull.

**Theorem 20 (message-rate bounds).** On $[0,T]$:

1. The number of pull/reinstate cycles per side is at most $T/c_{\min}+1$, whatever the signal.
2. With constant thresholds, it is also at most $\mathrm{TV}_{[0,T]}(\tau)/\Delta_\tau+1$, where TV is the total variation.
3. With requote throttling (cancel/replace only if the rounded price moves by at least one tick, and no more often than every $c_{\mathrm{rq}}$), the number of requotes per side is at most $T/c_{\mathrm{rq}}+1$.

*Proof.* (1) Cycles are disjoint in time and each lasts at least $c_{\min}$. (2) Within each cycle $\tau$ travels from at least $\theta_{\mathrm{on}}$ down to at most $\theta_{\mathrm{on}}-\Delta_\tau$. (3) Consecutive requotes are at least $c_{\mathrm{rq}}$ apart. $\square$

This rules out order-message thrashing by construction rather than by tuning.

### 8.5 Rounding and hard constraints

* The ask is rounded up and the bid down to the tick grid.
* Orders are post-only: $p^a\ge b_{\mathrm{best}}+\text{tick}$ and $p^b\le a_{\mathrm{best}}-\text{tick}$.
* If $q\ge q_{\max}$ the bid is not quoted; if $q\le-q_{\max}$ the ask is not quoted.
* Orders take effect only after the placement latency, and cancels after $\mathcal L_{\mathrm{rt}}$. Phase 5 models both explicitly, which is why $H\ge\mathcal L_{\mathrm{rt}}$ (Section 7.3).

### 8.6 Global kill switch

Both sides are pulled if any of the following holds:

* $\tilde\rho(t)\ge\rho_{\mathrm{crit}}$ (endogenous cascade, Section 6.3);
* either venue's feed has been silent for more than $T_{\mathrm{stale}}$;
* the out-of-order rejection rate exceeds its threshold (Section 4.5);
* any guard counter for NaN or Inf inputs increments.

Reinstatement follows the hysteresis rule of Section 8.4.

---

## 9. Per-event algorithm (specification level, not code)

Input: an event with integer timestamp $T_k$, channel $i=d_k$ and raw features. Steps marked † run only when $\Delta>0$. A heartbeat with no events runs steps 2–3, the ridge replenishment of step 7, and step 8 (Proposition 9 needs the heartbeat replenishment; spec 0.1.0 omitted it from this list). Book updates and status events do not step the model; their signals are computed at their time from the post-event state decayed read-only.

1. **Validate.** The timestamp, size and prices are finite, the size is positive and the channel is valid. Otherwise reject and count. Apply the out-of-order policy of Section 4.5.
2. † **Flush** the pending buffer into $R^{q,(0)}$. This gives the state at $t_{\mathrm{prev}}+$.
3. † **Interval update** with $\Delta=10^{-9}(T_k-T_{\mathrm{prev}})$:
   1. Compute $e_q$, with the cutoff of Section 4.6, and $d=e^{-\kappa\Delta}$.
   2. Compensators $\mathbf Z(\Delta)$ by Theorem 3, then $\boldsymbol\Lambda\mathrel{+}=\hat{\boldsymbol\Theta}\mathbf Z$.
   3. Forgetting: $\mathbf U\leftarrow\mathbf U/\sqrt d$, $\mathbf C\leftarrow d\,\mathbf C$, $\boldsymbol\omega^{\mathrm{eff}}\leftarrow d\,\boldsymbol\omega^{\mathrm{eff}}$.
   4. Build $\mathbf V$ from $R$, then $\mathbf F$, then the pivoted Cholesky factorization. For each row $\boldsymbol\ell_r$: Potter update with $\mathbf w_r=\mathbf V^\top\boldsymbol\ell_r$, then $\tilde{\boldsymbol\Theta}\leftarrow\tilde{\boldsymbol\Theta}-(\tilde{\boldsymbol\Theta}\mathbf w_r)\mathbf k_r^\top$.
   5. Decay $R$ by Theorem 2.
   6. Set $T_{\mathrm{prev}}\leftarrow T_k$.
4. **Regressor** $\mathbf z(t_k-)$ from $R$, excluding the pending buffer. **Mark** $g_k$ from the pre-update mark statistics, then update those statistics.
5. **Residual and CUSUM** for channel $i$: $\xi=\Lambda_i-\Lambda_i^{\mathrm{last}}$. On an alarm, set $\kappa$ for subsequent intervals to $\kappa_{\mathrm{fast}}$ until $t_k+\tau_{\mathrm{dwell}}$.
6. **Event increment:** $\mathbf C_{i,:}\mathrel{+}=\mathbf z^\top$ and $\tilde\theta_i\mathrel{+}=(\mathbf U\mathbf U^\top\mathbf z)^\top$.
7. **Ridge replenishment** of coordinate $p=\text{counter}\bmod P$ (Section 5.4), then increment the counter. If the interval of step 3 was longer than $T_{\mathrm{hb}}$, all coordinates were restored at its end (long-gap rule). **Row refresh** of row $k\bmod D$ (square-root form) or the Cholesky solve (information form).
8. **Serve:** clip $\hat\mu$; one Perron step for $\mathbf v$; project each excitation row; compute $\tilde\rho$; one Jacobi sweep for $\boldsymbol\lambda^\infty$.
9. **Pending:** $\pi_i\mathrel{+}=g_k$.
10. **Signals at $t_k+$**, using $R+\boldsymbol\pi$ read-only: $\mathbf m^{(0)}$, $\bar{\mathbf m}^{(0)}$, $\tau^\pm$, $\mathbf y^\infty$, $\mathbf X^\infty$ (Neumann, $K$ terms), $\hat\Delta$, $\sigma^2$, quotes, pull state machine and kill switch.

---

## 10. Verification obligations for Phase 3

Every recursion has a brute-force reference. Tolerances are relative and use double precision. The last column is the Phase 3 result: the worst error measured by the named test in `tests/cpp/` (release build, Apple M4; the same tests pass on GCC 13, Clang 18 with ASan, UBSan and TSan, and AppleClang in CI).

| # | Quantity | Brute-force reference | Tolerance | Phase 3 result |
|---|---|---|---|---|
| V1 | $R^{q,(n)}_j(t_k-)$, Erlang orders 1–2, with ties | direct $O(N)$ sum per evaluation ($O(N^2)$ total) | 1e-10 | 1.4e-15 (`KernelState.RecursionMatchesDirectSumsWithTies`) |
| V2 | compensator increments $\mathbf Z(\Delta)$, $\Delta\in[10^{-9},10^2]$ s | 30-point Gauss–Legendre on geometric pieces | 1e-10 | 1.3e-15 (`KernelState.CompensatorMatchesQuadrature`) |
| V3 | $\psi_n$, $n\le3$ | 50-digit quadrature | 1e-14 | 1.7e-15 (`Psi.MatchesFiftyDigitQuadrature`) |
| V4 | interval moment increment (Theorem 7) | quadrature of $\mathbf z\mathbf z^\top$ with $\mathbf z$ by direct sums | 1e-10 | passes for $\Delta$ from 1 ns to 60 s, both forms (`IntervalGram.MatchesQuadratureAndRankOneForm`) |
| V5 | $\tilde{\boldsymbol\Theta}$ online against batch (Theorem 10), with ties, marks and $\kappa$ switches | batch rebuild from the definitions (Python, `research/phase3/crosscheck_reference.py`) | 1e-10 | 2.9e-14 information form; 8.3e-12 Potter (`CrossCheck.*`) |
| V6 | $\mathbf U\mathbf U^\top\mathbf M-\mathbf I$ over $10^7$ events | $\mathbf M$ accumulated directly | 1e-8 | 6.3e-11 (`Estimator.PotterFactorStaysInverseOfMomentMatrix`) |
| V7 | row projection | sorting algorithm plus KKT residual | 1e-12 | 5.6e-16 against sorting; KKT 1.2e-15 (`Projection.*`) |
| V8 | Collatz–Wielandt bound $\ge\rho$ | lower Collatz–Wielandt bound at a power-iterated vector | exact inequality | holds on 3,000 random matrices (`CollatzWielandt.BoundsSpectralRadius`) |
| V9 | $\mathbf m^{(0)}$, $\bar{\mathbf m}^{(0)}$, $\mathbf y^\infty$, $\mathbf X^\infty$ | direct sums of survival functions; dense solve | 1e-10 | 2.7e-16, 4.8e-16, 1.8e-14, 4.6e-16 (`Forecast.MatchesDirectSumsAndDenseSolve`) |
| V10 | non-stationary recovery $\mathbf A_1\to\mathbf A_2$ | known ground truth (statistical test) | as in Section 5.8 | $D=4$, full bank: $\hat G$ 0.116 before (true 0.10), 0.515 after (true 0.50); lag to midpoint 17 s with the detector, 60 s without (`Regime.*`) |
| V11 | stress: $10^4$ events in 1 ms, $10^5$ same-timestamp events, $\Delta=1$ ns, $10^4$ s and $10^6$ s gaps, extreme marks, NaN/Inf/invalid inputs, out-of-order streams | invariants: finiteness, positive definiteness, $\hat{\mathbf G}\mathbf v\le\rho_{\max}\mathbf v$, no state change on rejection | exact | all hold, both back-ends (`Stress/*`, `StressModel.*`) |

The Python cross-check also compares every signal of a 496-event stream (intensities, toxicity, criticality, reservation shift, adverse-selection premia, variance) to 1.9e-12 for the information form. It matches every quote, flag and status exactly. The square-root form reaches 1.3e-9 on the cancellation-sensitive reservation shift, because its covariance-form rounding is amplified there (Section 5.4.1). That is one of the two reasons the information form is the default.

## 11. Assumptions and edge cases

### 11.1 Modeling assumptions

1. **Linear, excitation-only Hawkes.** Inhibition (for example sells suppressing buys) cannot be represented with $\hat\alpha\ge0$. A negative entry in the unconstrained $\tilde{\boldsymbol\Theta}$ is reported as a diagnostic of unmodeled inhibition rather than served.
2. **Fixed kernel bank** and **fixed mark coefficients** (Sections 2.3 and 3.2). Both are re-fit offline on a slow schedule.
3. **Parameters at evaluation time** (Section 3.2).
4. **Piecewise-stationary regimes**, changing more slowly than the fast half-life.
5. **Unit-mean conditional marks** (S3) for the default stability guarantee.
6. **A merged, time-ordered stream** with residual disorder of at most $\varepsilon_{\mathrm{ooo}}$, and cross-venue clock skew corrected upstream. Uncorrected skew larger than the Erlang modes would be absorbed into the lead-lag weights and bias them.
7. **Control approximations**: (A1)–(A5), CARA utility, first order in $\gamma J$, frozen inventory, exponential fill probability, and toxicity identified with the probability of informed flow.
8. **Impact parameters** $\eta_i$, $J^b$ and $J^I$ calibrated offline and stationary within a session.

### 11.2 Edge cases and their handling

| Case | Handling |
|---|---|
| $\Delta=0$ (ties) | batch semantics, order-invariant (Proposition 4) |
| $\Delta<0$ | clamp within $\varepsilon_{\mathrm{ooo}}$, otherwise reject with no state change (Proposition 5) |
| $\Delta$ tiny, down to 1 ns | $\psi$ series, exact to 2e-16 (Q2, Q3); interval rank adapts (Q4) |
| $\Delta$ huge (hours) | cutoff zeroes the states; the ridge restores $\mathbf M$, so $\tilde{\boldsymbol\Theta}$ moves toward the prior; signals decay to the stationary values |
| Same-timestamp burst of $10^5$ events | pending buffer accumulates; states finite (Section 4.6); no allocation |
| Channel silent for minutes | its column falls back to the prior with no wind-up (Proposition 9); heartbeat keeps $W_P$ bounded |
| NaN, Inf, size $\le0$, crossed or missing book | reject; for a missing book only, set $\iota=0$; guard counter |
| Unconstrained $\tilde\lambda\le0$ | never used where positivity matters; served $\hat\lambda\ge\mu_{\min}>0$ |
| Data look supercritical ($\tilde\rho\ge1$) | projection active, certified $\rho(\hat{\mathbf G})\le\rho_{\max}$; kill switch if $\tilde\rho\ge\rho_{\mathrm{crit}}$ |
| Reducible $\hat{\mathbf G}$ (zero rows or columns) | $\epsilon_v\mathbf v_{\mathrm{ref}}$ keeps $\mathbf v>0$; the certificate stays valid |
| Outlier prints | clipped, log-concave marks (Lemma 1) |
| Start-up | $\tilde{\boldsymbol\Theta}=\boldsymbol\Theta_0$; signals flagged "warming up" until the effective sample size exceeds a threshold |
| $\kappa\ge\beta_{\min}$ | rejected at configuration time (I2) |
| Large $D$ (8–12) | $P=33$–$49$, the matrices still fit in L1/L2 cache; cost grows like $D^2$; the second-core estimator option is recommended |

---

## 12. Decisions for the Phase 1 review

1. **Accept the discounted least-squares estimator as the core contribution**, with online EM and recursive Newton maximum likelihood kept as Phase 5 baselines (for example as ablations of strategy 5). This departs from the more common choice of likelihood-based online methods. Section 5.8 is the evidence for the change.
2. **Quoting venue.** The default is spot (B), with perp (A) as the leading signal. Quoting both venues is possible but doubles the control state.
3. **Initial channel set.** The default is $D=4$, with the $D=8$ depletion channels added in Phase 5.
4. **Estimator placement.** Inline on the hot thread (simplest; deterministic replay) or on a second core with seqlock publication. This can be decided in Phase 2 once there is a cost model.
5. **Default hyperparameters** in Appendix B, especially the kernel bank, the half-lives and $\rho_{\max}$.

---

## 13. References

* Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance* 8(3).
* Bacry, E., Bompaire, M., Gaïffas, S. and Muzy, J.-F. (2020). Sparse and low-rank multivariate Hawkes processes. *Journal of Machine Learning Research* 21.
* Bacry, E., Mastromatteo, I. and Muzy, J.-F. (2015). Hawkes processes in finance. *Market Microstructure and Liquidity* 1(1).
* Bierman, G. J. (1977). *Factorization Methods for Discrete Sequential Estimation*. Academic Press.
* Brémaud, P. and Massoulié, L. (1996). Stability of nonlinear Hawkes processes. *Annals of Probability* 24(3).
* Brown, E. N., Barbieri, R., Ventura, V., Kass, R. E. and Frank, L. M. (2002). The time-rescaling theorem and its application to neural spike train data analysis. *Neural Computation* 14(2).
* Cappé, O. and Moulines, E. (2009). On-line expectation–maximization algorithm for latent data models. *Journal of the Royal Statistical Society B* 71(3).
* Cartea, Á., Jaimungal, S. and Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
* Cont, R., Kukanov, A. and Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics* 12(1).
* Daley, D. J. and Vere-Jones, D. (2003). *An Introduction to the Theory of Point Processes, Vol. I*, 2nd ed. Springer.
* Duarte, A., Löcherbach, E. and Ost, G. (2019). Stability, convergence to equilibrium and simulation of non-linear Hawkes processes with memory kernels given by the sum of Erlang kernels. *ESAIM: Probability and Statistics* 23.
* Easley, D., López de Prado, M. and O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *Review of Financial Studies* 25(5).
* Filimonov, V. and Sornette, D. (2012). Quantifying reflexivity in financial markets: toward a prediction of flash crashes. *Physical Review E* 85.
* Glosten, L. R. and Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics* 14(1).
* Guéant, O., Lehalle, C.-A. and Fernandez-Tapia, J. (2013). Dealing with the inventory risk: a solution to the market making problem. *Mathematics and Financial Economics* 7(4).
* Hansen, N. R., Reynaud-Bouret, P. and Rivoirard, V. (2015). Lasso and probabilistic inequalities for multivariate point processes. *Bernoulli* 21(1).
* Hardiman, S. J., Bercot, N. and Bouchaud, J.-P. (2013). Critical reflexivity in financial markets: a Hawkes process analysis. *European Physical Journal B* 86.
* Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika* 58(1).
* Horn, R. A. and Johnson, C. R. (2013). *Matrix Analysis*, 2nd ed. Cambridge University Press.
* Kulhavý, R. and Zarrop, M. B. (1993). On a general concept of forgetting. *International Journal of Control* 58(4).
* Ogata, Y. (1981). On Lewis' simulation method for point processes. *IEEE Transactions on Information Theory* 27(1).
* Ozaki, T. (1979). Maximum likelihood estimation of Hawkes' self-exciting point processes. *Annals of the Institute of Statistical Mathematics* 31.
* Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41(1/2).
* Robbins, H. and Monro, S. (1951). A stochastic approximation method. *Annals of Mathematical Statistics* 22(3).
* Verhaegen, M. H. (1989). Round-off error propagation in four generally-applicable, recursive, least-squares estimation schemes. *Automatica* 25(3).

---

## Appendix A. Validation summary (quick mode)

`python research/phase1/validate_math_spec.py` runs these checks; all pass.

| Check | What it verifies | Error | Tolerance |
|---|---|---|---|
| Q1 | Erlang-1/2/3 recursion against $O(N^2)$, with ties | 5.6e-16 | 1e-12 |
| Q2 | $\psi$-form compensators (the naive form reaches 3.5e-9) | 2.2e-16 | 1e-14 |
| Q3 | $\psi_n$, $n\le3$ | 5.2e-16 | 5e-14 |
| Q4 | low-rank interval moment increment | 5.2e-16 | 1e-12 |
| Q5 | online estimate equals batch (Theorem 10) | 6.8e-14 | 1e-10 |
| Q6 | Potter factor after $10^5$ updates, $\operatorname{cond}=10^{12}$ | 8.0e-13 | 1e-6 |
| Q7 | Collatz–Wielandt bound; projection KKT | 1.1e-16; 1.6e-15 | 1e-12 |
| Q8 | Campbell means under Erlang-2 Hawkes (Monte Carlo) | 2.5e-3 | 3e-2 |

## Appendix B. Default hyperparameters (to be calibrated in Phase 5)

These are starting values, not results.

| Group | Parameter | Default |
|---|---|---|
| Kernel bank | $(r_q,\beta_q)$ | (1, 1000), (2, 200), (2, 50), (1, 1) per second |
| Marks | $c_u,c_L,c_\iota$; $u_{\max}$, $L_{\max}$; $\eta_g$, $\eta_\nu$ | 0.5, 0.5, 0.5; $10^4$, 50; 0.01, 0.01 |
| Forgetting | slow / fast half-life; dwell | 60 s / 5 s; 10 s |
| CUSUM | $\varrho$; $b_{\mathrm{cs}}$ | 1.5; 14 (set by in-control run-length simulation) |
| Ridge | $T_\omega$; heartbeat $T_{\mathrm{hb}}$ | 1 s; 100 ms |
| Stability | $\rho_{\max}$; $\chi$; $\epsilon_v$; $\rho_{\mathrm{crit}}$ | 0.97; 1.01; $10^{-6}$; 1.0 |
| Bounds | $\mu_{\min}$, $\mu_{\max}$ | $10^{-4}$ /s, $10^{6}$ /s |
| Numerics | $X_{\mathrm{cut}}$; $\psi$ series radius and terms; pivot $\eta$; $I_b$; $K$; $\varepsilon_{\mathrm{ooo}}$ | 50; 1 and 22; $10^{-15}$; 64; 3; 100 µs |
| Toxicity | $H$ | 50 ms |
| Control | $\gamma$, $k$, $T_h$, $\delta_{\max}$, $J^b$, $J^I$, $\Delta_\tau$, $c_{\min}$, $c_{\mathrm{rq}}$ | calibrated in Phase 5; $\Delta_\tau=0.1$, $c_{\min}=250$ ms, $c_{\mathrm{rq}}=5$ ms as starting points |

## Appendix C. Changes in spec version 0.2.0

Phase 3 implemented the spec and verified it against definition-level references. These changes resulted; none alters a theorem's statement.

1. **Long-gap rule (Section 5.4).** An interval longer than $T_{\mathrm{hb}}$ ends with a full ridge restore. Proposition 9's bound becomes unconditional, and the square-root form refactors $\mathbf U$ from $\mathbf M$ instead of applying Potter updates to a covariance inflated by $d^{-1}$. That would lose about 25 digits after a $10^4$ s gap.
2. **Heartbeats (Section 9).** A heartbeat performs the ridge replenishment that Proposition 9 assumes. Version 0.1.0's step list omitted it.
3. **Overflow-free Gram entry (Section 4.2).** $F_{00}$ is evaluated as $\Delta\,\psi_0(\kappa\Delta)$, so no positive exponent is ever formed.
4. **Information-form back-end (Section 5.4.1).** This form is added and made the default after measurement. The square-root form remains implemented and verified.
5. **Projection (Proposition 14).** An exact breakpoint search replaces the 64 bisection steps: no iteration count, no tolerance, and less work for $n=DQ\le64$.
6. **CUSUM batch rule (Section 5.5).** Resets take effect when a tie batch closes, which makes the detector order-invariant within a batch.
7. **Erratum (Section 7.2).** In the general formula for $\bar k_q(H)$ the power of $\beta_q$ is $r_q-1-l$, not $l$. The Erlang-1 and Erlang-2 special cases were already correct.
8. **Variance term (Theorem 19).** $\mathbb E[J_i^2]$ is defined under (A2).
