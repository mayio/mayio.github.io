---
layout: post
title: 'MASDA for Sparse Stereo Matching'
subtitle: What the uniqueness constraint is worth, measured against ground truth
thumbnail-img: https://raw.githubusercontent.com/mayio/mayio.github.io/master/assets/img/2026-08-07-MASDA-for-Sparse-Stereo-Matching_files/associations_periodic.png
date: '2026-08-07 19:00:00 +0200'
categories: association
comments: false
mathjax: true
author: Mario Lüder
---

In [Faster Data Association with Max-Sum Loopy Belief Propagation
(MASDA)](https://www.mariolueder.com/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA/)
I derived MASDA for the tracking problem: associate measurements with objects,
allowing for clutter and misdetection. Stereo matching has the same structure, so
this post applies MASDA there and measures what it gains.

Stereo is convenient for this. The one-to-one constraint is physically real, since a
surface point projects once into each image. Ground truth is obtainable, so I can
ask whether a match is *correct* rather than whether it looks plausible. And there
is a strong baseline: the exact linear assignment problem, solved by
Jonker-Volgenant.

Results, briefly:

- MASDA reaches the exact LAP optimum on all three test problems.
- Against mutual nearest-neighbour with a ratio test, it finds 2.75× more correct
  matches on ambiguous texture, and 2-3% more on easy texture.
- Precision collapses for every method on ambiguous texture, including the exact
  solver. Uniqueness is real information but it is not sufficient information.
- Adding an ordering factor is cheap and it works, in that crossings drop. It does
  not improve accuracy, because a bounded disparity range has already removed almost
  all of them.
- The speed advantage depends entirely on representation. A dense implementation is
  slower than scipy's Jonker-Volgenant; the same algorithm on an edge list is
  208-285× faster.

Everything runs from one script with no input images, so there is nothing to
license, and the disparity is known exactly.

---

## 1. Stereo as a data association problem

In tracking, measurements $$i \in \{1 \dots m\}$$ are associated with objects
$$j \in \{1 \dots n\}$$, at most one each way, with the option of calling a
measurement clutter or an object misdetected.

For stereo, substitute the nouns. Keypoints in the left image are measurements,
keypoints in the right image are objects. At most one association each way, because
one surface point produces one projection per image. A left keypoint whose surface
is occluded in the right view has no partner at all, which is clutter ($$\lambda$$).
A right keypoint whose partner the left detector missed is a misdetection
($$\gamma$$).

Occlusion is not a small correction here. In the scene below, 30% of left keypoints
have no attainable match, either because the surface is hidden or because the
corresponding right keypoint was never detected. A formulation that assumes every
keypoint is matchable is wrong a third of the time.

What stereo adds is geometry. The pair is rectified, so correspondences lie on the
same image row and disparity $$d = x_L - x_R$$ is positive and bounded. That makes the
association graph very sparse, which turns out to matter a lot (§6).

### 1.1 Factor graph

Same as the tracking case: binary association variables $$c_{ij}$$, clutter
indicators $$e_i$$, misdetection indicators $$\delta_j$$, similarity factors $$S_{ij}$$,
clutter factors $$\Lambda_i$$, misdetection factors $$\Gamma_j$$, and the exclusivity
constraints $$I_i$$ and $$E_j$$.

The graph is loopy. Every $$c_{ij}$$ sits in both an $$I_i$$ and an $$E_j$$ constraint, so
there are four-cycles everywhere. Hence loopy belief propagation, and hence no
convergence guarantee.

### 1.2 Messages

Unchanged:

$$
\begin{aligned}
\beta_{ij} &= s(i,j) - \max_{k \neq i} \rho_{kj} \\
\rho_{ij}  &= s(i,j) - \max_{k \neq j} \beta_{ik}
\end{aligned}
$$

with the non-association options competing inside those maxima:

$$
\begin{aligned}
\rho_{ij}  &= s(i,j) - \max\!\left(\lambda,\; \max_{k \neq j} \beta_{ik}\right) \\
\beta_{ij} &= s(i,j) - \max\!\left(\gamma,\; \max_{k \neq i} \rho_{kj}\right)
\end{aligned}
$$

$$\rho_{ij}$$ reads as: how good is associating $$i$$ with $$j$$, after subtracting the
best thing $$i$$ could do instead, where "instead" includes being called clutter.
$$\beta_{ij}$$ is the same from the object side.

Damping on both:

$$
x^{(t+1)} \leftarrow (1-\eta)\, x_{\text{target}} + \eta\, x^{(t)}
$$

The belief combines both directions,

$$
b_{ij} = \alpha_{ij} + \eta_{ij} + s_{ij}
$$

and since $$\beta_{ij} = s_{ij} + \alpha_{ij}$$ and $$\rho_{ij} = s_{ij} + \eta_{ij}$$,
in code this is

$$
b_{ij} = \beta_{ij} + \rho_{ij} - s_{ij}
$$

Both maxima exclude one element, so caching the largest and second-largest per row
and column makes "max excluding $$j$$" constant time. An iteration is then two linear
passes over the *edges*:

$$
O(T \cdot E), \qquad E = |\{(i,j) : \text{candidate}\}|
$$

In stereo $$E \approx 2.3\,m$$, so this is far smaller than $$O(T \cdot m \cdot n)$$.

### 1.3 Score, and the scale of $$\lambda$$ and $$\gamma$$

Descriptors are the Census transform over a $$7 \times 7$$ window: one bit per
neighbour, set when the neighbour is darker than the centre. That gives 48 bits,
which fit a `uint64`, so the distance is a single `popcount`. Census is invariant to
monotonic intensity mappings, which absorbs gain and offset differences between the
two sensors. Those differences are always present in practice.

Two unrelated Census descriptors agree on half their bits by chance, so scaling the
Hamming distance $$h$$ around that point gives a score with a usable zero:

$$
s(i,j) = \underbrace{\frac{B/2 - h(i,j)}{B/2}}_{\text{+1 perfect, 0 chance}}
       \;-\; w_y \left(\frac{y_i - y_j}{\sigma_y}\right)^{2}
$$

with $$B = 48$$. The second term penalises vertical residual: on a rectified pair a
true match has $$y_i = y_j$$. I keep it even though it is usually small, because the
median $$|\Delta y|$$ over accepted matches then doubles as a cheap online check on
calibration drift.

On this scale $$\lambda = \gamma = -0.1$$ means "reject anything worse than a tenth of
the way from chance to perfect", which is easier to reason about than a tuned
constant.

---

## 2. A scene with known answers

Without a rangefinder, real stereo footage cannot tell you whether a match is
correct. Synthetic data can, so I generate the scene:

```python
def ground_truth_disparity():
    d = np.full((H, W), 8.0, np.float32)          # back wall, far
    yy, xx = np.mgrid[0:H, 0:W]
    floor = yy > H * 0.55
    d[floor] = 8.0 + (yy[floor] - H * 0.55) * 0.28  # slanted floor
    d[(xx > 300) & (xx < 400) & (yy > 90) & (yy < 250)] = 26.0   # box
    d[(xx > 150) & (xx < 162) & (yy > 40) & (yy < 300)] = 38.0   # thin bar
    return d
```

Each feature breaks something. The slanted floor is sampled at different rates by
the two views, so foreshortening violates one-to-one. The box edges generate
occlusion. The thin bar violates the ordering constraint that scanline methods rely
on.

The right image is a forward warp of the left by the true disparity, with a
z-buffer, so occlusions appear on their own rather than being modelled:

```python
for x, x2, d in zip(xs[ok], xr[ok], disp[y][ok]):
    if d > zbuf[y, x2]:          # nearer surface wins
        zbuf[y, x2] = d
        right[y, x2] = left[y, x]
```

![scene](https://raw.githubusercontent.com/mayio/mayio.github.io/master/assets/img/2026-08-07-MASDA-for-Sparse-Stereo-Matching_files/scene.png)

### 2.1 Three textures

The same geometry is rendered three ways, because texture rather than geometry
decides how hard the association is:

| texture | what it is | why |
|---|---|---|
| broadband | multi-scale noise | descriptors are individually discriminative; the easy case |
| dots | pseudo-random blobs | imitates an IR projector, as on a RealSense D435 |
| periodic | regular lattice | repetitive structure: brick, fencing, tiling. The hard case. |

---

## 3. Measuring the ambiguity

Descriptor degeneracy and matching ambiguity are different things, and only the
second one matters.

![descriptors](https://raw.githubusercontent.com/mayio/mayio.github.io/master/assets/img/2026-08-07-MASDA-for-Sparse-Stereo-Matching_files/descriptors.png)

Over roughly 1400-2000 keypoints per image:

| texture | distinct descriptors | median score margin | margin < 0.05 |
|---|---|---|---|
| broadband | 94% | 0.833 | 3.4% |
| dots | 93% | 0.792 | 6.2% |
| periodic | 74% | 0.083 | 41.2% |

The score margin is best candidate minus runner-up, per left keypoint. It is what
decides difficulty, and it is what a ratio test keys on. On the lattice the median
margin is ten times smaller and 41% of keypoints have effectively tied candidates.

The distinct-descriptor count is misleading. The dot texture has 93% distinct
descriptors, almost the same as broadband, yet this is supposed to be the ambiguous
case. Anti-aliasing and sensor noise make every window unique as a bit pattern while
leaving it uninformative as an identity. Counting distinct descriptors overstates
how much information they carry; the margin measures it directly.

---

## 4. Implementation

The solver, in the notation above:

```python
def masda(S, lam=-0.1, gam=-0.1, iters=30, damping=0.4, eps=1e-5):
    finite = np.isfinite(S)
    beta = np.where(finite, 0.0, -np.inf)
    rho  = beta.copy()

    for it in range(iters):
        # rho: a row's other options, or calling this measurement clutter
        comp = np.maximum(lam, top2_excluding(np.where(finite, beta, -np.inf), 1))
        new_rho = np.where(finite, (1 - damping) * (S - comp) + damping * rho, -np.inf)

        # beta: a column's other options, or calling this object misdetected
        comp = np.maximum(gam, top2_excluding(np.where(finite, new_rho, -np.inf), 0))
        new_beta = np.where(finite, (1 - damping) * (S - comp) + damping * beta, -np.inf)

        delta = max(np.nanmax(np.abs(new_rho - rho)),
                    np.nanmax(np.abs(new_beta - beta)))
        rho, beta = new_rho, new_beta
        if delta < eps:
            break

    belief = np.where(finite, beta + rho - S, -np.inf)
    return decide(S, belief, lam)
```

### 4.1 Reading out the answer

I got this wrong on the first attempt, and the mistake follows directly from the
theory, so it is worth walking through.

I gated acceptance on $$b_{ij} > 0$$. On problems with exactly tied candidates that
returned zero matches, on problems whose optimum matched everything.

The belief measures an edge's advantage *over its competitors*. When nothing has an
advantage, every belief is $$\le 0$$. That is also the condition under which the LP
relaxation has no unique optimum, and uniqueness is exactly what Bayati, Shah and
Sharma require for max-product to be correct on bipartite matching. So the
degenerate case is not an edge case to patch around; it is where the guarantee
stops.

> Bayati, M., Shah, D., & Sharma, M. (2008). *Max-Product for Maximum Weight
> Matching: Convergence, Correctness, and LP Duality.* IEEE Transactions on
> Information Theory, 54(3), 1241-1251.
> [doi:10.1109/TIT.2007.915695](https://doi.org/10.1109/TIT.2007.915695)

Two questions were tangled together:

| question | answered by |
|---|---|
| which candidate? | the belief $$b_{ij}$$, as an ordering. Its sign means nothing. |
| associate at all? | $$s(i,j)$$ against $$\lambda$$, which is what $$\lambda$$ is for. |

So: order by belief, decide by $$\lambda$$, require row and column to agree, then fill
in greedily by belief over what is left.

```python
def decide(S, belief, lam):
    order = np.argsort(-belief, axis=None)
    used_i, used_j, out = np.zeros(m, bool), np.zeros(n, bool), {}
    for flat in order:
        i, j = divmod(int(flat), n)
        if not np.isfinite(belief[i, j]):
            break
        if S[i, j] <= lam or used_i[i] or used_j[j]:
            continue
        out[i] = j
        used_i[i] = used_j[j] = True
    return out
```

The greedy fill is not cosmetic. Under near-ties every row's best belief points at
the same column, so requiring mutual agreement commits exactly one pair. Every
greedily accepted edge has $$s > \lambda$$ and two free endpoints, so it raises the
objective. Adding it moved agreement with exhaustive search from 56/60 to 58/60 on
problems small enough to enumerate.

---

## 5. Results against ground truth

$$\lambda = \gamma = -0.1$$, 30 iterations, damping 0.4. Recall is measured against
attainable matches only: a left keypoint counts in the denominator only if its true
correspondence is unoccluded *and* was itself detected in the right image. Counting
the rest would charge the matcher for the detector's misses.

### broadband

| method | matches | correct | wrong | precision | recall | objective |
|---|---|---|---|---|---|---|
| Mutual-NN + ratio | 774 | 670 | 104 | 0.866 | 0.800 | 464.58 |
| MASDA | 802 | 681 | 121 | 0.849 | 0.813 | 478.77 |
| Optimal LAP (JV) | 802 | 680 | 122 | 0.848 | 0.811 | 478.77 |

### dots

| method | matches | correct | wrong | precision | recall | objective |
|---|---|---|---|---|---|---|
| Mutual-NN + ratio | 840 | 689 | 151 | 0.820 | 0.818 | 492.98 |
| MASDA | 891 | 707 | 184 | 0.793 | 0.840 | 521.10 |
| Optimal LAP (JV) | 891 | 708 | 183 | 0.795 | 0.841 | 521.10 |

### periodic

| method | matches | correct | wrong | precision | recall | objective |
|---|---|---|---|---|---|---|
| Mutual-NN + ratio | 401 | 84 | 317 | 0.209 | 0.145 | −0.70 |
| MASDA | 1134 | 231 | 903 | 0.204 | 0.400 | 641.23 |
| Optimal LAP (JV) | 1134 | 204 | 930 | 0.180 | 0.353 | 647.15 |

![associations](https://raw.githubusercontent.com/mayio/mayio.github.io/master/assets/img/2026-08-07-MASDA-for-Sparse-Stereo-Matching_files/associations_periodic.png)

![comparison](https://raw.githubusercontent.com/mayio/mayio.github.io/master/assets/img/2026-08-07-MASDA-for-Sparse-Stereo-Matching_files/comparison.png)

### 5.1 Reading the numbers

**MASDA reaches the optimum.** Objective ratios against exact Jonker-Volgenant are
1.0000, 1.0000 and 0.9909. Loopy max-sum, with no guarantee available, lands on the
LAP optimum on all three problems. That is not what "approximate inference on a
loopy graph" leads you to expect.

On the lattice it also gets *more correct matches* than the exact solver, 231
against 204, while scoring slightly lower on the objective. The objective is a
proxy, and maximising it exactly does not maximise correctness.

**The gain over nearest-neighbour tracks ambiguity.** Correct matches:

| texture | median margin | MASDA | Mutual-NN | ratio |
|---|---|---|---|---|
| broadband | 0.833 | 681 | 670 | 1.02× |
| dots | 0.792 | 707 | 689 | 1.03× |
| periodic | 0.083 | 231 | 84 | 2.75× |

Where descriptors discriminate, uniqueness adds two or three percent and a ratio
test is a perfectly reasonable matcher. Where they do not, MASDA finds 2.75× more
correct correspondences. Note mutual-NN's objective on the lattice: −0.70, negative.
The ratio test rejects so much that it pays more in clutter and misdetection cost
than it earns in matches. It is not trading badly; it is declining to trade.

**Precision collapses for everyone.** 0.204 for MASDA, 0.209 for mutual-NN, 0.180
for the exact optimum.

This is the result I find most useful. Uniqueness is real information and it is
being used optimally here, since the exact solver does no better. But on genuinely
repetitive texture the information is not in the descriptors, and constraint
propagation cannot create it. What MASDA does is convert a refusal to answer into
answers, most of which are wrong. Whether that helps depends on the consumer: a
bundle adjustment with a robust loss will happily take 231 good matches out of 1134,
while a naive triangulation will be poisoned.

So the claim is not that MASDA beats nearest-neighbour. It is that MASDA extracts
everything the uniqueness constraint contains, which is a lot when descriptors are
ambiguous and not enough when they are degenerate. That is what motivates adding
further factors, which is §7.

### 5.2 Damping

![damping](https://raw.githubusercontent.com/mayio/mayio.github.io/master/assets/img/2026-08-07-MASDA-for-Sparse-Stereo-Matching_files/damping.png)

Undamped max-sum on the ambiguous problem does not settle; the largest message
change plateaus instead of decaying. Damping of 0.3-0.5 stabilises it, and solution
quality is flat across that range, staying within a fraction of a percent of
optimal.

On real data (a D435 IR pair, 848×480, around 1100 keypoints) the messages never
formally converge at any iteration budget, yet the *decision* is stable to four
significant figures from 50 iterations and within 0.1% at 20. The oscillation is
confined to messages that do not change the answer. Message convergence is not the
property you actually need.

---

## 6. Speed: the representation decides it

The complexity argument is $$O(T \cdot E)$$ against Jonker-Volgenant's $$O(N^3)$$. My
first implementation did not benefit from it at all.

Written the obvious way, with messages in an $$m \times n$$ array padded with
$$-\infty$$, MASDA is slower than scipy's compiled Jonker-Volgenant:

| texture | nodes | edges | dense MASDA | JV (scipy) | ratio |
|---|---|---|---|---|---|
| broadband | 1382 | 3263 | 2648 ms | 1834 ms | 0.7× |
| dots | 1601 | 3769 | 3267 ms | 2007 ms | 0.6× |
| periodic | 2019 | 4507 | 7403 ms | 3388 ms | 0.5× |

With $$m \approx n \approx 1400$$ the matrix holds about two million cells and 3263
real edges, so roughly 600× of the arithmetic goes into entries that are $$-\infty$$.
The $$O(T \cdot E)$$ bound assumes an edge list. A dense array gives
$$O(T \cdot m \cdot n)$$ and vectorisation does not recover the difference.

### 6.1 Edge-list formulation

Put the messages on the edges. The only awkward part is that both updates need
$$\max_{k \neq j}$$ over a row or column, which is quadratic in row length if done
directly. Three segment reductions answer it exactly in $$O(E)$$:

```python
def _seg_max_excluding(vals, idx, n):
    """Per-segment max with each element's own contribution removed, in O(E)."""
    m1 = np.full(n, -np.inf)
    np.maximum.at(m1, idx, vals)          # segment max
    at_max = vals >= m1[idx]              # m1 is the max, so >= means ==
    cnt = np.zeros(n, np.int64)
    np.add.at(cnt, idx[at_max], 1)        # how many attain it
    m2 = np.full(n, -np.inf)
    below = ~at_max
    if below.any():
        np.maximum.at(m2, idx[below], vals[below])   # max strictly below
    second = np.where(cnt > 1, m1, m2)
    return np.where(at_max, second[idx], m1[idx])
```

Three cases: an element below the max sees the max; an element at the max also sees
the max, provided something else attains it; otherwise it sees the runner-up. Ties
need handling rather than ignoring, since near-ties are the case of interest.

The solver is then five lines per iteration:

```python
def masda_sparse(ei, ej, se, m, n, lam=-0.1, gam=-0.1, iters=30, damping=0.4):
    beta = np.zeros(len(se)); rho = np.zeros(len(se))
    for _ in range(iters):
        comp = np.maximum(lam, _seg_max_excluding(beta, ei, m))
        new_rho  = (1 - damping) * (se - comp) + damping * rho
        comp = np.maximum(gam, _seg_max_excluding(new_rho, ej, n))
        new_beta = (1 - damping) * (se - comp) + damping * beta
        rho, beta = new_rho, new_beta
    belief = beta + rho - se
    ...
```

Same messages, same damping, same belief, same decision rule. Only the
representation differs.

### 6.2 What that gains

| texture | edges | dense | sparse | JV | vs dense | vs JV |
|---|---|---|---|---|---|---|
| broadband | 3263 | 2648 ms | 8.8 ms | 1834 ms | 300× | 208× |
| dots | 3769 | 3267 ms | 9.3 ms | 2007 ms | 351× | 216× |
| periodic | 4507 | 7403 ms | 11.9 ms | 3388 ms | 624× | 285× |

208-285× faster than compiled Jonker-Volgenant, from interpreted NumPy.

Quality is unchanged. Objectives match the dense solver to four decimals in all
three cases, and correct-match counts differ by at most three out of several
hundred, where the two orderings break ties differently: 681 against 682, 707
against 708, and 231 against 228. Both remain at the LAP optimum. The assignments are not bit-identical,
and with tied beliefs they need not be.

As an independent check, the same algorithm as a C++ edge-list implementation on
real imagery (848×480 IR pair, about 1075 keypoints, 2882 candidate edges, 20
iterations) runs in 1.67 ms against a 33.3 ms frame budget at 30 Hz. The keypoint
detector, at 21 ms, costs more than ten times as much.

### 6.3 The actual claim

MASDA's cost is linear in the number of *plausible* associations, and in a
geometrically constrained problem that is a small fraction of $$m \times n$$. Here the
epipolar band and disparity range cut roughly two million possible pairings down to
3300 candidates. Only a representation that exploits that sees any benefit.

This also reframes the comparison with an exact solver. It is not about accuracy,
since Jonker-Volgenant is exactly as good and occasionally slightly better. It is
that MASDA is anytime and incremental: it can be stopped early with a usable answer,
its messages carry over between frames when the problem changes slightly, and it
accepts factors that destroy the assignment structure, where a LAP solver cannot
follow.

---

## 7. Can MASDA express the ordering constraint?

Scanline stereo methods use the ordering constraint: matches along a scanline should
not cross. Plain MASDA cannot express it, which is a standing objection to using it
for stereo. It can be added as a factor, the derivation is tidier than I expected, and it
turns out not to be worth using.

Two associations $$(i,j)$$ and $$(i',j')$$ cross iff $$(x_i - x_{i'})(x_j - x_{j'}) < 0$$.
A matching is order-preserving exactly when no two of its pairs cross, so ordering
decomposes into *pairwise* factors with no higher-order term. That is what makes it
tractable.

Take $$\psi(c_e, c_f) = -\kappa$$ when both edges are on and crossing, else 0. For a
pairwise factor between binary variables only the difference of the outgoing message
matters, and with $$\mu_f = m_f(1) - m_f(0)$$:

$$
\Delta_{\psi \to e} = \max(0, \mu_f - \kappa) - \max(0, \mu_f)
                    = -\operatorname{clamp}(\mu_f,\, 0,\, \kappa)
$$

The ordering message is the conflicting edge's own preference, clamped and negated.
One scalar, constant time. I checked it against brute-force max-sum over 20000
random cases; agreement is 4×10⁻¹⁶.

Because these messages are additive on the edge they fold into the score. Writing
$$o_e = -\sum_{f \in X(e)} \operatorname{clamp}(b_f, 0, \kappa)$$ for the summed
ordering pressure, the updates are the same two reductions with $$s + o$$ substituted
for $$s$$. Nothing new has to be maintained. Crossing pairs depend only on geometry,
so they are computed once; $$O(E_r^2)$$ per scanline band here, and a Fenwick tree
over the sorted $$j$$ order would make it $$O(E_r \log E_r)$$ if bands got dense.

$$\kappa$$ stays finite. Thin foreground objects genuinely violate ordering, and a
hard constraint would delete them. Damping goes up to 0.6, because these factors add
loops that the bipartite convergence result does not cover.

### 7.1 It works, and it does not help

The lattice is an unfair test, because repetitive mistakes tend to be
order-preserving: a region shifted by one period crosses nothing. So I built a scene
for it instead, with nine thin bars at assorted depths over broadband texture, where
errors do cross.

| $$\kappa$$ | matches | correct | precision | crossings |
|---|---|---|---|---|
| off | 711 | 506 | 0.712 | 67 / 2790 |
| 0.1 | 712 | 505 | 0.709 | 64 |
| 0.4 | 712 | 503 | 0.706 | 59 |
| 0.8 | 711 | 503 | 0.707 | 58 |

The factor does what it is supposed to do: crossings fall by 13%, monotonically in
$$\kappa$$. Correct matches fall by three. On the lattice the effect is much larger in
the same direction, crossings 229 → 112 at $$\kappa = 0.3$$, and accuracy again drops
slightly, 226 → 216 correct.

So the constraint is enforced and the answer gets marginally worse. Both scenes
agree on that, which is the outcome I did not expect and the one worth explaining.

### 7.2 Why it does not help

The baseline has 67 crossings out of 2790 same-band pairs, 2.4%. There was very
little for the constraint to fix, and the crossings it does remove are apparently
not the wrong matches.

That is not a property of the scene. Matches $$(i,j)$$ and $$(i',j')$$ with
$$x_i < x_{i'}$$ cross iff $$x_j > x_{j'}$$, that is

$$
d_{i'} - d_i \;>\; x_{i'} - x_i
$$

A crossing requires the disparity difference to exceed the horizontal separation.
With disparities confined to a range of width $$d_{\max} - d_{\min}$$, crossings are only
possible between keypoints closer together in $$x$$ than that width, and get rarer as
the range tightens.

So the disparity-range gate already does most of ordering's work. Uniqueness plus a
bounded disparity range gives largely ordered solutions for free, and the few
crossings left over are as likely to be correct as not: the nine thin bars in this
scene violate ordering *legitimately*, and a factor that penalises crossings cannot
tell those apart from crossings caused by mismatches.

Ordering is therefore expressible, cleanly, inside the existing closed form, and not
worth switching on in a geometrically gated sparse matcher. I would still expect it
to pay where the gating is weak: a wide disparity range, an uncalibrated pair, or
two-dimensional temporal association, where nothing constrains ordering for free.
Here it costs accuracy to buy a statistic nobody consumes.

---

## 8. Comparison with existing work

**Jonker-Volgenant / Hungarian.** Exact, $$O(N^3)$$, and here at least as good as
MASDA everywhere. For a pure assignment problem of moderate size, use it. MASDA
earns its place when you intend to add factors that stop the problem being a LAP.

> Jonker, R., & Volgenant, A. (1987). *A shortest augmenting path algorithm for
> dense and sparse linear assignment problems.* Computing, 38(4), 325-340.
> [doi:10.1007/BF02278710](https://doi.org/10.1007/BF02278710)

**SPADA / sum-product data association.** Produces marginal association
probabilities rather than a MAP assignment, at higher cost. If the consumer wants
soft weights, for a PDA-style tracker or a differentiable pipeline, that is the
right choice. Stereo wants a decision per keypoint, so MAP is what is needed.

**Sinkhorn and optimal transport, as in SuperGlue.** Structurally very close: soft
one-to-one assignment with dustbins, which are $$\lambda$$ and $$\gamma$$ under another
name. Sinkhorn is the entropy-regularised relaxation and max-sum is its
zero-temperature limit. SuperGlue's real advantage is that its scores come from a
learned attention network instead of a hand-designed $$s(i,j)$$, and its dustbin costs
are learned rather than set. That points straight at the weakest part of what I have
here.

> Sarlin, P.-E., DeTone, D., Malisiewicz, T., & Rabinovich, A. (2020).
> *SuperGlue: Learning Feature Matching with Graph Neural Networks.* CVPR.
> [arXiv:1911.11763](https://arxiv.org/abs/1911.11763)

**Semi-global matching and dense stereo.** A different problem, and worth saying why
sparse matching is not simply the worse option. Dense stereo assigns a disparity to
every pixel with a smoothness prior on the pixel grid; per scanline, one-to-one with
occlusion is solved exactly by dynamic programming in $$O(W \cdot D)$$, and DP also
encodes the ordering constraint that MASDA needs a factor for. If you want a dense
disparity map, MASDA is the wrong tool. Sparse matching is the right tool when you
want a few hundred well-localised sub-pixel correspondences to feed geometry, such
as odometry, calibration or structure, rather than a depth image.

> Hirschmüller, H. (2008). *Stereo Processing by Semiglobal Matching and Mutual
> Information.* IEEE TPAMI, 30(2), 328-341.
> [doi:10.1109/TPAMI.2007.1166](https://doi.org/10.1109/TPAMI.2007.1166)

ELAS is worth mentioning too: it uses a triangulated set of robustly matched support
points as a prior for dense estimation, which is close to the sparse-then-densify
arrangement a MASDA front end would naturally feed.

> Geiger, A., Roser, M., & Urtasun, R. (2010). *Efficient Large-Scale Stereo
> Matching.* ACCV.

---

## 9. Advantages

Where MASDA is the right choice:

- Cost is linear in plausible associations rather than in $$m \times n$$. With
  geometric constraints cutting candidates to about 2.3 per keypoint, the sparse
  form runs 208-285× faster than an exact LAP solver at the same quality.
- It is optimal, or indistinguishable from optimal, on these problems without
  needing to be.
- It is anytime. A handful of iterations gives a usable answer, and the decision
  stabilises well before the messages do.
- It extends. Adding an ordering, smoothness or temporal factor keeps a factor graph
  a factor graph, whereas it stops being a LAP. §7 is the demonstration: a new
  pairwise constraint cost one clamped scalar per conflicting edge and no change to
  the update structure.
- Clutter and misdetection are first-class rather than post-hoc thresholds, which
  matters when occlusion makes 30% of keypoints unmatchable.

Where it is not:

- No convergence guarantee, and the guarantee that exists lapses precisely when the
  problem is ambiguous, which is when you wanted help.
- It cannot create information. On degenerate texture it produces confident wrong
  answers where a ratio test produces none, and which of those is preferable is a
  property of the consumer.
- $$\lambda$$ and $$\gamma$$ are hand-set. The scale here is interpretable, which helps,
  but that is not the same as calibrated.

---

## 10. What would improve the matcher

Ranked by how much I expect them to matter.

**Better scores.** $$s(i,j)$$, $$\lambda$$ and $$\gamma$$ are the weakest part of this by
a wide margin. Everything in §5 says the constraint machinery is already extracting
what it can, and the shortfall is in the evidence being fed to it. A small model
over descriptor distance, vertical residual, response ratio and local texture energy,
trained against ground truth to output a calibrated log-likelihood ratio, would
change the numbers more than any refinement of the message passing. SuperGlue's
results point the same way.

**Disparity smoothness over a neighbourhood graph.** Neighbouring keypoints on the
same surface should have similar disparity, and nothing here uses that. It is the
largest piece of information currently ignored. Before deriving new messages the
cheap version is worth trying: match, fit a robust disparity surface over a Delaunay
triangulation of the left keypoints, rescore, rematch. I have tried a
nearest-neighbour-graph version of this and it did not help, but the prior was fitted
from the same matches it then judged, which makes it close to self-confirming. A
properly triangulated version with a robust cost is a different experiment.

**Sub-pixel disparity.** Depth accuracy depends on this more than on anything else.
Keypoint positions are refined sub-pixel, but disparity is currently just the
difference of two independently refined positions rather than a fit to the
correlation surface between them.

**Unrolled training.** Whether $$T$$ max-sum iterations can be unrolled with a
soft-max relaxation and trained end to end. Structurally that is what SuperGlue does
with Sinkhorn in the zero-entropy limit, so the interesting question is whether the
max-sum version is competitive with fewer parameters.

**Temporal association.** The same machinery for frame-to-frame matching, for
ego-motion. Two things that failed for stereo should pay there: the search is
two-dimensional so ordering does not come free from a disparity range, and $$k$$ is
genuinely large so a motion prior from IMU-derived rotation compensation has
something to prune.

A caveat on all of the above. These numbers come from one synthetic scene. I expect
the qualitative pattern to hold generally: the gain over a ratio test tracks
ambiguity, loopy max-sum lands on the LAP optimum, precision collapses under
degeneracy for every method, and ordering is redundant once disparity is bounded. The
specific figures are not a benchmark.
