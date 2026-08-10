---
layout: post
title: 'Dense Stereo Matching with Max-Sum Belief Propagation on Sparse Matrices (MASDA)'
subtitle: 'The one-to-one constraint applied to every pixel: what uniqueness buys over winner-take-all, how close loopy max-sum gets to the exact assignment optimum, and why the sparse-matrix representation decides everything.'
thumbnail-img: /assets/img/2026-08-08-Dense-MASDA_files/maps_teddy.png
date: '2026-08-07 19:00:00 +0200'
categories: association
comments: false
mathjax: true
author: Mario Lüder
tags: [belief-propagation, data-association, computer-vision]
---

In [Faster Data Association with Max-Sum Loopy Belief Propagation
(MASDA)](https://www.mariolueder.com/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA/)
I derived [MASDA](#masda) for the tracking problem: associate measurements with
objects, allowing for [clutter and misdetection](#clutter-and-misdetection). Stereo
matching has the same structure, so this post applies MASDA there — to the *dense*
problem, a [disparity](#disparity) for every pixel — and measures what it gains.

The formulation runs on **sparse matrices**. Each pixel offers only its two best
disparity candidates out of the aggregated [cost volume](#cost-volume), so the
association graph carries two edges per pixel instead of a $$W \times W$$ matrix
per row, and every
result below depends on that representation twice over: it is what makes the exact
comparison feasible, and it is what makes the solver fast.

*[Part 2](https://www.mariolueder.com/2026-08-08-Dense-MASDA-Belief-Propagation-Stereo-on-a-Jetson/)
takes this formulation into C++ on a Jetson TX2 and measures it against SGM;
[Part 3](https://www.mariolueder.com/2026-08-09-Realtime-Dense-MASDA-on-the-Jetson-GPU/)
makes it real-time on the TX2's GPU, bit-identically.*

*Every technical term this series uses — from factor graphs to CUDA warps — is
defined in the [appendix](#appendix-terms-concepts-and-sources), with links to the
original work. Terms link there on first use, in all three parts.*

Results, briefly — pooled over eight [Middlebury](#middlebury-stereo-datasets)
scenes with [structured-light ground truth](#structured-light-ground-truth), roughly
1.3 million answers:

- **The [one-to-one constraint](#one-to-one-constraint) is worth +10.7 points of
  [precision](#coverage-precision-and-the-bad-pixel-rate) over
  [winner-take-all](#winner-take-all) on identical scores**: 0.884 against 0.776,
  while keeping 96% of WTA's correct answers. Per scene the gain is 8–13 points,
  largest where the texture is worst.
- **Loopy [max-sum](#max-sum-max-product-and-sum-product) matches the exact
  [assignment](#linear-assignment-problem) optimum on precision** where the
  exact optimum is computable: per-row
  [Jonker-Volgenant](#jonker-volgenant-and-the-hungarian-method) reaches 0.914 and
  0.941 on Teddy and Cones; MASDA reaches 0.915 and 0.942.
- It does that from **two message-passing iterations**, sitting 0.8% short of the
  optimal [objective](#objective-ratio) with **1.6% of rows exactly optimal**.
  Thirty iterations close the objective gap to 0.08% and make 47–69% of rows
  optimal — and return *no extra precision at all*. The decision stabilises an
  order of magnitude before the objective does, which is the single most useful
  thing I learned here.
- **The representation decides the speed.** The same messages on dense per-row
  matrices: 7.2 s per frame. On sparse matrices: **0.38 s — 19×** — and 20× faster
  than compiled per-row Jonker-Volgenant. The engineered C++ descendant of this
  NumPy study solves the same frames in ~11 ms, and the full GPU pipeline of
  Part 3 runs at 28.9 ms per frame end to end.

Everything is regenerable: `article/dense_sparse_matrices.py` produces every
number in this post from the shipping binary's own cost volume, so the NumPy
study and the production pipeline are measured on identical scores.

---

## 1. Dense stereo as a data association problem

In tracking, [data association](#data-association) puts measurements
$$i \in \{1 \dots m\}$$ with objects $$j \in \{1 \dots n\}$$, at most one each way,
with the option of calling a measurement clutter or an object misdetected.

For stereo, substitute the nouns — per
[rectified](#rectification-and-epipolar-geometry) image row. Pixels in the left
row are measurements, pixels in the right row are objects. At most one
association each way, because one surface point produces one projection per
image. A left pixel whose surface is occluded in the right view has no partner
at all, which is clutter ($$\lambda$$). A right pixel whose surface is hidden
from the left camera is a misdetection ($$\gamma$$). [Occlusion](#occlusion) is not
a small
correction: on these scenes it affects 10–20% of pixels, and a formulation that
assumes every pixel is matchable is wrong exactly at the depth discontinuities
where stereo is hardest.

What stereo adds is geometry. The pair is rectified, so correspondences lie on
the same image row and disparity $$d = x_L - x_R$$ is positive and bounded. Each
row is an independent association problem — which makes the whole thing
parallel, and later, a GPU workload.

### 1.1 Factor graph

Same as the tracking case, on the same [factor graph](#factor-graph): binary
association variables $$c_{ij}$$, clutter
indicators $$e_i$$, misdetection indicators $$\delta_j$$, similarity factors $$S_{ij}$$,
clutter factors $$\Lambda_i$$, misdetection factors $$\Gamma_j$$, and the exclusivity
constraints $$I_i$$ and $$E_j$$.

The graph is loopy. Every $$c_{ij}$$ sits in both an $$I_i$$ and an $$E_j$$ constraint, so
there are four-cycles everywhere. Hence [loopy belief
propagation](#loopy-belief-propagation), and hence no
[convergence guarantee](#convergence-and-the-anytime-property).

### 1.2 Messages

The [two message directions](#messages-responsibility-and-availability) are
unchanged:

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

[Damping](#damping) on both:

$$
x^{(t+1)} \leftarrow (1-\eta)\, x_{\text{target}} + \eta\, x^{(t)}
$$

The [belief](#belief) combines both directions,

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

In the dense formulation, $$E = 2$$ per pixel — the two best candidates out of the
aggregated volume — so a 450-pixel row carries ~900 edges instead of the ~200,000
cells of its full matrix. Why two is the right number is measured in Part 2: keeping
eight candidates per pixel is measurably *worse* than keeping two, because the extra
candidates are noise the solver has to argue with.

### 1.3 The score

Descriptors are the [Census transform](#census-transform) over a
$$7 \times 7$$ window: one bit per
neighbour, set when the neighbour is darker than the centre. 48 bits fit a
`uint64`, so the distance is a single
[`popcount`](#hamming-distance-and-popcount), and Census is invariant to
monotonic intensity mappings, which absorbs gain and offset differences between
two real sensors.

Two unrelated Census descriptors agree on half their bits by chance, so scaling
the [Hamming distance](#hamming-distance-and-popcount) $$h$$ around that point
gives a score with a usable zero:
$$+1$$ perfect, $$0$$ chance. On this scale $$\lambda = \gamma = -0.1$$ means
"reject anything worse than a tenth of the way from chance to perfect", which is
easier to reason about than a tuned constant.

One pixel's Census comparison quantises to only 49 levels, which is not enough
signal per pixel. The score MASDA actually consumes is
[*aggregated*](#cost-aggregation) over an
edge-aware support region — an [O(N) recursive
filter](#edge-aware-recursive-filter) that stops at intensity
edges — so each candidate's score summarises a neighbourhood while respecting
depth boundaries. The aggregation machinery, and the measurements behind each of
its choices, are Part 2's subject; here it is the given: the sparse candidate
matrix is built from the aggregated volume of the shipping implementation, so
this study and the production pipeline score identical evidence.

---

## 2. Ground truth

The matcher is measured on **Teddy** and **Cones** from the [Middlebury
2003 stereo set](#middlebury-stereo-datasets), and on the six Middlebury 2005
scenes: Art, Books, Dolls, Laundry,
Moebius and Reindeer, at third size. They are rectified, and they ship
[structured-light ground truth](#structured-light-ground-truth) at quarter-pixel
resolution with a few percent of
pixels marked unknown. The [dataset page](https://vision.middlebury.edu/stereo/data/)
states: "We grant permission to use and publish all images and disparity maps on
this website."

> D. Scharstein and R. Szeliski (2003). *High-accuracy stereo depth maps using
> structured light.* CVPR, 195-202.
> [doi:10.1109/CVPR.2003.1211354](https://doi.org/10.1109/CVPR.2003.1211354)

> D. Scharstein and C. Pal (2007). *Learning conditional random fields for stereo.*
> CVPR.
> [doi:10.1109/CVPR.2007.383191](https://doi.org/10.1109/CVPR.2007.383191)

The 2005 two-view archives ship no documented disparity scale, so the factor of 3
is established rather than assumed: for a pixel at $$x$$ in the left view with true
disparity $$t$$, the right view's disparity map at $$x - t$$ must also read $$t$$.
That identity holds to a median of 0.000 px at a scale of 3 and fails at every
other integer, and it involves no matcher, so it cannot flatter the results.

Middlebury marks unknown disparity as zero rather than shipping a separate
visibility mask, so an answer landing on an unknown pixel has no correct value to
be compared against. Those are excluded from precision rather than scored as
wrong, which would charge the matcher for holes in the dataset.

---

## 3. Implementation

The solver, in the notation above — the dense-matrix reference version:

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

### 3.1 Reading out the answer

I got this wrong on the first attempt, and the mistake follows directly from the
theory, so it is worth walking through.

I gated acceptance on $$b_{ij} > 0$$. On problems with exactly tied candidates that
returned zero matches, on problems whose optimum matched everything.

The belief measures an edge's advantage *over its competitors*. When nothing has an
advantage, every belief is $$\le 0$$. That is also the condition under which
[the LP relaxation has no unique
optimum](#lp-relaxation-and-the-uniqueness-condition), and uniqueness is exactly
what Bayati, Shah and
Sharma require for [max-product](#max-sum-max-product-and-sum-product) to be
correct on bipartite matching. So the
degenerate case is not an edge case to patch around; it is where the guarantee
stops. Section 4 measures how often that happens on dense rows, and it is often.

> Bayati, M., Shah, D., & Sharma, M. (2008). *Max-Product for Maximum Weight
> Matching: Convergence, Correctness, and LP Duality.* IEEE Transactions on
> Information Theory, 54(3), 1241-1251.
> [doi:10.1109/TIT.2007.915695](https://doi.org/10.1109/TIT.2007.915695)

Two questions were tangled together:

| question | answered by |
|---|---|
| which candidate? | the belief $$b_{ij}$$, as an ordering. Its sign means nothing. |
| associate at all? | $$s(i,j)$$ against $$\lambda$$, which is what $$\lambda$$ is for. |

So: [order by belief, decide by $$\lambda$$](#greedy-decode), require row and column
to agree, then fill
in greedily by belief over what is left. The greedy fill is not cosmetic: under
near-ties every row's best belief points at the same column, so requiring mutual
agreement commits exactly one pair, and every greedily accepted edge has
$$s > \lambda$$ and two free endpoints, so it raises the objective.

### 3.2 The sparse-matrix form — this is the design, not an optimisation

Put the messages on the edges. The only awkward part is that both updates need
$$\max_{k \neq j}$$ over a row or column, which is quadratic in row length if done
directly. Three [segment reductions](#segment-reduction-and-max-excluding) answer it
exactly in $$O(E)$$:

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

Same messages, same damping, same belief, same decision rule as the dense-matrix
version. Only the representation differs — and section 5 shows that difference is
a factor of 62 on this problem, before any compiled code.

A note on [damping](#damping): undamped max-sum on heavily tied problems does not
settle; the
largest message change plateaus instead of decaying. Damping of 0.3–0.5 stabilises
it, and solution quality is flat across that range. [Message convergence is not the
property you need](#convergence-and-the-anytime-property) anyway — the *decision*
stabilises long before the messages do,
and [section 4.3](#43-two-iterations-against-thirty) measures how much sooner:
fifteen times sooner, on these scenes.

---

## 4. Results against ground truth

$$\lambda = \gamma = -0.1$$, **two iterations** — the shipping configuration —
damping 0.4, candidates = top-2 per pixel from the aggregated volume, tolerance
1 px, unknown ground truth excluded. ([Section 4.3](#43-two-iterations-against-thirty)
measures what the other 28 iterations would buy, which is nothing.)
Three solvers on **identical scores**: [winner-take-all](#winner-take-all) (argmax
over the full volume — no uniqueness), MASDA on the sparse candidate matrix, and
per-row exact
assignment ([Jonker-Volgenant](#jonker-volgenant-and-the-hungarian-method) with
explicit $$\lambda$$/$$\gamma$$ slots, so
non-association is a real option for it too).

### 4.1 What uniqueness is worth

Pooled over all eight scenes — about 1.3 million answered pixels per method:

| method | correct | wrong | precision |
|---|---|---|---|
| winner-take-all | 252,567 | 72,838 | 0.776 |
| **MASDA, sparse matrices** | 242,824 | 31,992 | **0.884** |

**+10.7 points of precision for the one-to-one constraint, at the cost of 3.9% of
the correct answers.** WTA answers everywhere and is wrong more than twice as
often. Per scene:

| scene | WTA | MASDA | Δ |
|---|---|---|---|
| Art | 0.700 | 0.830 | +13.0 |
| Books | 0.796 | 0.901 | +10.5 |
| Dolls | 0.810 | 0.915 | +10.5 |
| Laundry | 0.661 | 0.777 | +11.6 |
| Moebius | 0.788 | 0.878 | +9.0 |
| Reindeer | 0.769 | 0.905 | +13.6 |
| Cones | 0.859 | 0.942 | +8.3 |
| Teddy | 0.826 | 0.911 | +8.5 |

The gain is largest on the worst scenes (Art, Laundry, Reindeer), which is the
right shape: where the descriptor evidence is weakest, mutual exclusivity has the
most to contribute. This is the dense-problem version of the original
[keypoint study's](#keypoints-and-detector-repeatability) central finding — the
constraint pays in proportion to the [ambiguity](#repetitive-texture-and-ambiguity) —
now measured on three orders of magnitude more answers.

For the full engineered pipeline (the C++ implementation with its
[margin gate](#margin-and-the-margin-gate),
which trades a little coverage for precision), the comparison against OpenCV's
[SGM](#semi-global-matching) lands at
**9.7% [bad-1.0](#coverage-precision-and-the-bad-pixel-rate) against SGM's 10.9%**
at 76% versus 78% [coverage](#coverage-precision-and-the-bad-pixel-rate) —
Part 2's headline, reproduced here only for orientation.

![dense maps](/assets/img/2026-08-08-Dense-MASDA_files/maps_teddy.png)

### 4.2 Against the exact optimum

Per-row Jonker-Volgenant is feasible on the sparse problem (900 edges per row),
so the exact comparison covers every row of Teddy and Cones:

| scene | method | correct | precision | objective ratio | rows at exact optimum |
|---|---|---|---|---|---|
| Teddy | MASDA, 2 iters | 130,413 | **0.915** | 0.9918 | 6 / 369 |
| Teddy | MASDA, 30 iters | 131,152 | 0.914 | 0.9992 | 175 / 369 |
| Teddy | JV (exact) | 131,210 | 0.914 | 1 | — |
| Cones | MASDA, 2 iters | 130,956 | **0.942** | 0.9939 | 0 / 369 |
| Cones | MASDA, 30 iters | 131,443 | 0.941 | 0.9996 | 253 / 369 |
| Cones | JV (exact) | 131,482 | 0.941 | 1 | — |

Three readings, and the third is the one I did not expect.

**Precision is indistinguishable from exact.** 0.915 against 0.914, 0.942 against
0.941. The exact solver finds a few hundred more correct answers out of 131
thousand and is fractionally *less* precise, because it answers more often.
Whatever approximation error loopy max-sum commits here, it is not made of wrong
disparities.

**It is genuinely an approximation now.** On the original keypoint problems,
MASDA's [objective ratio](#objective-ratio) against JV was 1.0000 — it simply *was*
optimal, three
problems out of three. Here it never is: 0.9918 at the shipping setting, and even
at thirty iterations it reaches the exact optimum on only 47% and 69% of rows. The
difference is ties. An aggregated Census volume quantises to few enough levels
that exactly tied candidates are everywhere, so the
[LP-uniqueness condition](#lp-relaxation-and-the-uniqueness-condition) of the
Bayati-Shah-Sharma guarantee fails routinely.

**And the gap costs nothing.** Two iterations sit 0.8% short of the optimal
objective with 1.6% of rows optimal; thirty iterations close that to 0.08% and 47%
— and return no extra precision whatsoever. The objective is a proxy, and this is
the cleanest evidence I have that it is a loose one: an 0.8% objective deficit and
a 30× shortfall in exactly-solved rows are invisible in the only quantity a
consumer experiences.

### 4.3 Two iterations against thirty

Since the shipping implementation runs two iterations and this study originally
ran thirty, the difference is worth its own table. Pooled over all eight scenes:

| iterations | correct | precision | objective ratio (Teddy) | rows optimal |
|---|---|---|---|---|
| **2** (shipping) | 242,824 | **0.884** | 0.9918 | 6 / 369 |
| 30 | 243,589 | 0.880 | 0.9992 | 175 / 369 |

Fifteen times the message passing buys 0.3% more correct answers, 0.4 points
*less* precision, and a solve four times slower. The extra iterations move
marginal candidates from "not answered" to "answered", and those marginal
candidates are wrong slightly more often than the population average — so the
objective improves while precision does not.

This is the quantitative form of a claim [section 3.2](#32-the-sparse-matrix-form--this-is-the-design-not-an-optimisation)
makes qualitatively: **[message convergence is not the property you
need](#convergence-and-the-anytime-property).** The
messages are still moving at iteration thirty; the decision stopped moving around
iteration two. Every number elsewhere in this article is therefore reported at the
shipping setting, and I would treat any belief-propagation matcher quoting an
iteration count without this comparison with suspicion — including my own earlier
version of this page.

---

## 5. Speed: the representation decides it

The complexity argument is $$O(T \cdot E)$$ against
[Jonker-Volgenant's](#jonker-volgenant-and-the-hungarian-method)
$$O(N^3)$$. Written the obvious way — messages in a per-row $$W \times W$$ matrix
padded with $$-\infty$$ — the argument buys nothing: with two real candidates per
pixel, more than 99% of the arithmetic lands on $$-\infty$$ cells.

One frame of Teddy (369 rows, 326,565 edges), same solvers, same answers:

| representation | 2 iterations | 30 iterations |
|---|---|---|
| dense per-row matrices | 7.22 s | 96.9 s |
| **sparse matrices (edge list)** | **0.38 s** | 1.57 s |
| per-row JV (scipy, compiled) | 7.6 s | 10.0 s |

At the shipping setting the sparse-matrix solver is **19× faster than the same
mathematics on dense matrices and 20× faster than compiled exact assignment** —
from interpreted NumPy. (Jonker-Volgenant does not depend on the iteration count
at all; its two entries differ only by measurement noise on a shared desktop,
which is a useful reminder of how much precision to read into any single row of
this table.) These are study numbers, not production numbers: the identical algorithm
in C++ (Part 2) solves a frame in ~11 ms on a desktop and ~23 ms on the Jetson
TX2's ARM cores, and the full GPU pipeline (Part 3) — cost volume, aggregation
and solve together — runs at 28.9 ms per frame at 848×480, bit-identical to the
C++ output. Five orders of magnitude between the first table's first row and the
shipping pipeline, and not one change to the messages: representation,
then engineering.

The actual claim, then: MASDA's cost is linear in the number of *plausible*
associations, and the aggregated volume plus geometry cuts a $$W \times W$$ row
problem to two candidates per pixel. Only a representation that exploits that
sees any benefit. This also reframes the comparison with an exact solver: it is
not about accuracy — section 4.2 shows JV is exactly as good — it is that MASDA
is [anytime](#convergence-and-the-anytime-property), incremental, and accepts
factors that destroy the assignment
structure, where a [LAP solver](#linear-assignment-problem) cannot follow.

---

## 6. Can MASDA express the ordering constraint?

Scanline stereo methods use the [ordering constraint](#ordering-constraint): matches
along a scanline
should not cross. This is the one thing scanline dynamic programming gets for
free and a plain assignment formulation does not, so it is the standing
objection to using MASDA for dense stereo. It can be added as a factor, the
derivation is tidier than I expected, and in the dense formulation it behaves
differently from how it behaved on keypoints.

Two associations $$(i,j)$$ and $$(i',j')$$ cross iff
$$(x_i - x_{i'})(x_j - x_{j'}) < 0$$. A matching is order-preserving exactly when
no two of its pairs cross, so ordering decomposes into *pairwise* factors with no
higher-order term. That is what makes it tractable.

Take $$\psi(c_e, c_f) = -\kappa$$ when both edges are on and crossing, else 0. For
a pairwise factor between binary variables only the difference of the outgoing
message matters, and with $$\mu_f = m_f(1) - m_f(0)$$:

$$
\Delta_{\psi \to e} = \max(0, \mu_f - \kappa) - \max(0, \mu_f)
                    = -\operatorname{clamp}(\mu_f,\, 0,\, \kappa)
$$

The ordering message is the conflicting edge's own preference, clamped and
negated. One scalar, constant time. I checked it against brute-force max-sum over
20000 random cases; agreement is 4×10⁻¹⁶.

**It composes with the sparse-matrix form without touching it.** Because these
messages are additive on the edge they fold into the score: writing
$$o_e = -\sum_{f \in X(e)} \operatorname{clamp}(b_f, 0, \kappa)$$ for the summed
ordering pressure, the updates are the same two segment reductions of
[section 3.2](#32-the-sparse-matrix-form--this-is-the-design-not-an-optimisation)
with $$s + o$$ substituted for $$s$$. No new message type, no change to the
solver's structure. This is the property that makes MASDA worth preferring over
an exact assignment solver, which cannot follow here at all.

$$\kappa$$ stays finite. Thin foreground objects genuinely violate ordering, and
a hard constraint would delete them — the standard failure of DP-based scanline
methods, which is why they need forbidden-move exceptions.

### 6.1 Measured, dense: it works, and it costs more than it returns

Eight scenes, every fourth row, paired against the same rows with the factor off,
$$\kappa = 0.3$$, damping raised to 0.6 because the ordering factors add loops
the bipartite convergence result does not cover:

| scene | precision, off | on | Δ | crossings retained |
|---|---|---|---|---|
| Art | 0.825 | 0.828 | +0.3 | 0.67× |
| Books | 0.897 | 0.901 | +0.4 | 0.33× |
| Dolls | 0.910 | 0.911 | +0.1 | 0.56× |
| Laundry | 0.771 | 0.780 | +0.9 | 0.47× |
| Moebius | 0.874 | 0.879 | +0.5 | 0.65× |
| Reindeer | 0.900 | 0.907 | +0.7 | 0.60× |
| Cones | 0.941 | 0.943 | +0.2 | 0.59× |
| Teddy | 0.912 | 0.915 | +0.3 | 0.47× |

**The factor does exactly what the derivation says: it removes about half the
crossings (0.54× on average, never worse than 0.67×) and improves precision on
all eight scenes** — mean +0.43 points, standard error 0.094, and unanimous,
which matters more than the $$t$$-statistic on eight scenes. The gain is largest
where the scene is worst (Laundry +0.9 at precision 0.771, Reindeer +0.7), the
same shape uniqueness itself shows in [section 4.1](#41-what-uniqueness-is-worth).

And it is **not worth switching on**, for a reason that is specific to the dense
formulation:

| | edges per row | crossing pairs per row | solve time |
|---|---|---|---|
| keypoints (original study) | ~30 | tens | negligible |
| **dense, sparse matrices** | ~885 | **~2400** | **5–7× slower** |

In the keypoint configuration, crossing pairs were a footnote — the scanline band
held a handful of keypoints and the $$O(E_r^2)$$ enumeration cost nothing. In the
dense configuration *the band is the whole row*: every pixel is a node, and
crossing pairs outnumber edges by nearly 3:1. Enumerating and reducing over
600,000 of them per scene is 5–7× the entire solve, to buy 0.4 points of
precision — against a factor of 62 that the sparse-matrix representation buys for
free, and 40 more correct answers out of 243,000.

The $$\kappa$$ sweep says the same thing from another direction: 0.1, 0.3 and 0.8
all land within 0.003 of one another on precision and within 1% on crossings
retained. The factor saturates immediately — it is not being tuned into
usefulness, it is doing all it can do and that is a small thing.

So the honest position: **ordering is expressible, cleanly, inside the existing
closed form, and on the dense problem it is a real but poor trade.** The $$O(E_r
\log E_r)$$ [Fenwick-tree](#fenwick-tree) construction I waved at in the keypoint
version is now
the *precondition* rather than an optimisation, and even at that price the
0.4-point return does not obviously justify the loops it adds to a graph whose
convergence is already unguaranteed. Where I would expect it to matter is
precisely where the geometry stops helping: an uncalibrated pair, or
two-dimensional temporal association, where nothing constrains ordering for free.

One structural note worth keeping. Crossings correlate strongly with the error
rate rather than with the geometry — the scenes with the most crossings (Art
31,116; Laundry 27,126) are the two least precise, and Cones and Books, the most
precise, have the fewest. Crossings are largely a *symptom* of wrong matches, not
an independent property, which is why penalising them removes wrong matches and
why the effect is largest on the scenes that need it most. It also caps the
upside: you cannot fix an error the constraint cannot see, and a patch of
[repetitive texture](#repetitive-texture-and-ambiguity) matched one period off
crosses nothing at all.

## 7. Comparison with existing work

**[Jonker-Volgenant / Hungarian](#jonker-volgenant-and-the-hungarian-method).**
Exact, $$O(N^3)$$, and here exactly as good as
MASDA where it can be run at all ([section 4.2](#42-against-the-exact-optimum)).
For a pure assignment problem of moderate size, use it. MASDA earns its place on
speed at scale — 6.4× in NumPy, far more engineered — and when you intend to add
factors that stop the problem being a LAP.

> Jonker, R., & Volgenant, A. (1987). *A shortest augmenting path algorithm for
> dense and sparse linear assignment problems.* Computing, 38(4), 325-340.
> [doi:10.1007/BF02278710](https://doi.org/10.1007/BF02278710)

**[SPADA / sum-product data association](#spada).** Produces marginal association
probabilities rather than a MAP assignment, at higher cost. If the consumer wants
soft weights, that is the right choice. Stereo wants a decision per pixel, so MAP
is what is needed.

**[Sinkhorn and optimal transport, as in
SuperGlue](#sinkhorn-optimal-transport-and-superglue).** Structurally very close: soft
one-to-one assignment with dustbins, which are $$\lambda$$ and $$\gamma$$ under another
name. Sinkhorn is the entropy-regularised relaxation and max-sum is its
zero-temperature limit. SuperGlue's real advantage is that its scores come from a
learned attention network instead of a hand-designed $$s(i,j)$$, and its dustbin costs
are learned rather than set. That points straight at the weakest part of what I have
here.

> Sarlin, P.-E., DeTone, D., Malisiewicz, T., & Rabinovich, A. (2020).
> *SuperGlue: Learning Feature Matching with Graph Neural Networks.* CVPR.
> [arXiv:1911.11763](https://arxiv.org/abs/1911.11763)

**[Semi-global matching](#semi-global-matching).** The standard fast dense method,
and the direct
competitor now that this formulation is dense too. SGM aggregates
[smoothness](#smoothness-prior)
along scanline paths and has **no uniqueness constraint at all** — a
[left-right consistency check](#left-right-consistency-check) is bolted on
afterwards, which costs a second matcher run.
MASDA gets mutual exclusivity inside the inference, in one run, plus a per-pixel
confidence (the [margin](#margin-and-the-margin-gate)) as a by-product. Measured head-to-head in Part 2:
**9.7% bad-1.0 against SGM's 10.9%** over these eight scenes, at 76% against 78%
coverage. The two mechanisms are orthogonal, and the interesting object — a
factor graph with both uniqueness and path smoothness — does not exist in either
tool today.

> Hirschmüller, H. (2008). *Stereo Processing by Semiglobal Matching and Mutual
> Information.* IEEE TPAMI, 30(2), 328-341.
> [doi:10.1109/TPAMI.2007.1166](https://doi.org/10.1109/TPAMI.2007.1166)

**[ELAS](#elas)** narrows a dense search around triangulated support points — the
avoid-the-sweep family, alongside [PatchMatch](#patchmatch-stereo) and
[rSGM](#rsgm). Part 2 measured this project's version of that idea (a
[coarse-to-fine](#coarse-to-fine) mask) at accuracy parity and recorded exactly
where its speedup goes to die on embedded hardware.

> Geiger, A., Roser, M., & Urtasun, R. (2011). *Efficient Large-Scale Stereo
> Matching.* ACCV 2010, LNCS 6492, 25-38.
> [doi:10.1007/978-3-642-19315-6_3](https://doi.org/10.1007/978-3-642-19315-6_3)

---

## 8. Advantages

Where MASDA on sparse matrices is the right choice:

- Cost is linear in plausible associations rather than in $$m \times n$$. With two
  candidates per pixel, that is 19× over the same solver on dense matrices and
  20× over compiled exact assignment, before any engineering.
- It is indistinguishable from optimal on precision, while being measurably
  non-optimal on the objective — the useful direction of that trade.
- It is anytime, and [section 4.3](#43-two-iterations-against-thirty) puts a
  number on it: two iterations give the same precision as thirty, at a quarter of
  the cost. The decision stabilises an order of magnitude before the messages do,
  which is why the C++ implementation ships with two.
- It extends, and this is the main reason to prefer it over an exact
  [LAP solver](#linear-assignment-problem).
  Adding an [ordering](#ordering-constraint), [smoothness](#smoothness-prior) or
  temporal factor keeps a [factor graph](#factor-graph) a factor
  graph, whereas it stops being an assignment problem.
  [Section 6](#6-can-masda-express-the-ordering-constraint) is the demonstration
  *and* the caution: a new pairwise constraint costs one clamped scalar per
  conflicting edge and folds into the existing reductions — and on the dense
  problem the crossing-pair enumeration it needs outweighs what it returns.
- Clutter and misdetection are first-class rather than post-hoc thresholds, which
  matters when occlusion makes 10–20% of pixels unmatchable.

Where it is not:

- The correctness guarantee is conditional and the condition fails routinely on
  dense rows — half the rows here have non-unique LP optima. In practice that
  cost under 0.1% of objective and no measurable precision, but nothing in the
  theory promised it, and section 4.2 is the honest record.
- It cannot create information. Where the aggregated evidence is degenerate it
  produces confident wrong answers; the margin gate exists to convert those back
  into abstentions, at the price of coverage.
- $$\lambda$$ and $$\gamma$$ are hand-set. The scale here is interpretable, which
  helps, but that is not the same as calibrated.

---

## 9. What would improve it

**Better scores.** $$s(i,j)$$, $$\lambda$$ and $$\gamma$$ remain the weakest part.
Everything in section 4 says the constraint machinery extracts what the evidence
contains — MASDA equals exact inference — so the shortfall is in the evidence. A
small model trained against ground truth to output a calibrated log-likelihood
ratio would change these numbers more than any refinement of the message passing.

**A [smoothness factor](#smoothness-prior), done properly.** (Ordering is now
measured rather than
open — section 6 — and the interesting question it leaves is whether a
neighbourhood factor pays where ordering did not.) Neighbouring pixels on the same surface
have similar disparity, and the current factor graph ignores it. The cheap
variants are measured negatives (Part 2's record); the real derivation — path
aggregation as factors, so uniqueness and smoothness live in one graph — is the
interesting object this formulation makes possible.

**[Sub-pixel disparity](#sub-pixel-disparity).** The candidates are integer; the fit
to the correlation
surface between them is not done, and depth accuracy depends on it more than on
anything else downstream.

**Temporal factors.** The same machinery for frame-to-frame association, where
the previous frame's solution is a prior. A first prototype (frame $$t$$'s
disparities masking frame $$t{+}1$$'s candidates) already measures positive on
the engineered pipeline, and unlike a coarse pass, the prior is free.

A caveat on all of the above: eight scenes at 450×375 is not a benchmark, and the
specific figures are what this code did on these scenes. What I would expect to
generalise is the shape: uniqueness pays in proportion to ambiguity, loopy
max-sum matches exact inference on precision while violating its guarantee's
conditions, and the representation — not the mathematics — decides whether any
of it is usable.

---

## Appendix: terms, concepts and sources

This series moves between three fields — inference on graphical models, stereo
vision, and embedded performance work — and each brings its own vocabulary. Every
term the three parts use is defined here once, with links to the work it comes
from, so that no part assumes the others have been read.

Two conventions. Each term is linked to its entry here the first time it appears in
each part, and again where a later section turns on it; Parts 2 and 3 link back to
this page. And where an entry states something as *measured on this project* rather
than as established practice, it says so and names the section that measured it —
the distinction between "this is what the field knows" and "this is what my board
did" is the one I most want to keep visible.

### A.1 Inference on factor graphs

#### Factor graph

A bipartite graph with *variable* nodes and *factor* nodes, where each factor
connects the variables it scores and the whole graph represents one big function as
a sum (or product) of local terms. It is the data structure that makes message
passing possible: an algorithm can be written entirely as "what each node tells its
neighbours". The canonical reference is Kschischang, Frey and Loeliger; my own
worked introduction, with code, is the [loopy belief propagation
post](https://www.mariolueder.com/2022-09-17-Loopy-Belief-Propagation/#factor-graphs).

> F. R. Kschischang, B. J. Frey, H.-A. Loeliger (2001). *Factor graphs and the
> sum-product algorithm.* IEEE Transactions on Information Theory 47(2), 498-519.
> [doi:10.1109/18.910572](https://doi.org/10.1109/18.910572)

#### Loopy belief propagation

Belief propagation is exact on a factor graph that is a tree: one sweep in each
direction and every node knows its answer. Run the same local update rule on a
graph *with cycles* and you get loopy belief propagation — no guarantee of
convergence and no guarantee of correctness, but in practice often excellent. The
association graph here is loopy by construction: every association variable sits in
both a row constraint and a column constraint, so four-cycles are everywhere
([section 1.1](#11-factor-graph)). Weiss and Freeman give the theoretical account
of what a fixed point of the max-product version does and does not mean.

> Y. Weiss, W. T. Freeman (2001). *On the optimality of solutions of the
> max-product belief-propagation algorithm in arbitrary graphs.* IEEE Transactions
> on Information Theory 47(2), 736-744.
> [doi:10.1109/18.910585](https://doi.org/10.1109/18.910585)

Two earlier posts of mine build loopy BP from scratch — [on a
grid](https://www.mariolueder.com/2022-09-24-Noise-Reduction-Loopy-Belief-Propagation/)
for image denoising, and [in the Gaussian
case](https://www.mariolueder.com/2023-08-21-Line-Fitting-using-Gaussian-Loopy-Belief-Propagation/)
for line fitting.

#### Max-sum, max-product and sum-product

Three names for the same message-passing skeleton with different operators inside
it. *Sum-product* marginalises: it computes, for each variable, the probability of
each of its states. *Max-product* maximises instead of summing, so it seeks the
single best joint configuration. *Max-sum* is max-product in the log domain, where
products become sums — numerically better behaved, and the reason the messages in
[section 1.2](#12-messages) are additive. MASDA is the max-sum member of the
family; see [SPADA](#spada) for the sum-product one.

#### Messages: responsibility and availability

The two message directions. $$\rho_{ij}$$ flows from the measurement side and asks
"how good is this pair, net of the best thing $$i$$ could do instead"; $$\beta_{ij}$$
asks the same from the object side. The naming follows affinity propagation, where
$$\rho$$ is the *responsibility* a point takes for an exemplar and the
*availability* $$\alpha$$ is what the exemplar offers back — here
$$\beta_{ij} = s_{ij} + \alpha_{ij}$$, which is why the belief is
$$\beta + \rho - s$$. Frey and Dueck introduced the pair; Givoni and Frey's binary
variable model is the derivation the [original MASDA
post](https://www.mariolueder.com/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA/#masda)
follows.

> B. J. Frey, D. Dueck (2007). *Clustering by passing messages between data
> points.* Science 315(5814), 972-976.
> [doi:10.1126/science.1136800](https://doi.org/10.1126/science.1136800)

> I. E. Givoni, B. J. Frey (2009). *A Binary Variable Model for Affinity
> Propagation.* Neural Computation 21(6), 1589-1600.
> [doi:10.1162/neco.2009.05-08-785](https://doi.org/10.1162/neco.2009.05-08-785)

#### Belief

The score a variable ends up with once both message directions are combined:
$$b_{ij} = \beta_{ij} + \rho_{ij} - s_{ij}$$. It measures an edge's advantage over
its competitors, which is exactly why its *sign* carries no information about
whether to associate at all — the mistake [section 3.1](#31-reading-out-the-answer)
is about. The belief is used as an ordering; $$\lambda$$ decides membership.

#### Damping

Blending each new message with the previous one,
$$x^{(t+1)} \leftarrow (1-\eta) x_{\text{target}} + \eta x^{(t)}$$, to stop
oscillation. Standard practice in loopy BP and not optional here: undamped max-sum
on heavily tied problems plateaus instead of settling. Measured on this problem,
quality is flat for $$\eta$$ between 0.3 and 0.5, and everything in this series
runs at 0.4 (0.6 for the ordering experiment of [section 6.1](#61-measured-dense-it-works-and-it-costs-more-than-it-returns),
which adds loops).

#### MAP estimate

Maximum a posteriori: the single most probable joint configuration, as opposed to
the per-variable marginals. Stereo wants one disparity per pixel, so MAP is the
right target and max-sum is the right algorithm.

#### Convergence and the anytime property

*Convergence* here means the messages stop changing. *Anytime* means the algorithm
can be stopped at any iteration and still return a usable answer, improving with
time. The most useful thing this project measured is that the two are far apart:
the messages are still moving at iteration thirty while the decision stopped moving
around iteration two, so the shipping configuration runs two iterations
([section 4.3](#43-two-iterations-against-thirty)). Being anytime is also the
property an exact assignment solver lacks.

### A.2 The association problem

#### Data association

The problem of deciding *which observation belongs to which thing*: measurements to
tracked objects in a radar, detections to identities in a tracker, left pixels to
right pixels in stereo. What makes it a problem rather than a lookup is that the
answer is constrained jointly — one measurement cannot belong to two objects — so
the decisions cannot be made independently per measurement.

#### MASDA

Max-Sum Algorithm Data Association: max-sum loopy belief propagation on the
association factor graph, which is what this series is about. The derivation — the
factor graph, the exclusivity constraints, the closed-form messages and the
complexity argument — is in the [original MASDA
post](https://www.mariolueder.com/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA/).
This part reuses those equations unchanged and substitutes stereo's nouns
([section 1](#1-dense-stereo-as-a-data-association-problem)).

#### SPADA

Sum-Product Algorithm Data Association: the same factor graph solved with
sum-product, returning marginal association *probabilities* instead of one
assignment. The right choice when a downstream consumer wants soft weights, and
more expensive. Williams and Lau are the reference treatment; the comparison with
MASDA is [in the original
post](https://www.mariolueder.com/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA/#spada-sum-product-algorithm-data-association-and-masda-max-sum-algorithm-data-association).

> J. Williams, R. Lau (2014). *Approximate evaluation of marginal association
> probabilities with belief propagation.* IEEE Transactions on Aerospace and
> Electronic Systems 50(4), 2942-2959.
> [arXiv:1209.6299](https://arxiv.org/abs/1209.6299)

#### Clutter and misdetection

The two outside options that keep the association problem honest. *Clutter*
($$\lambda$$) is a measurement that belongs to nothing — in stereo, a left pixel
whose surface is not visible on the right. *Misdetection* ($$\gamma$$) is an object
that no measurement found — a right pixel hidden from the left camera. Both are
first-class variables in the factor graph rather than thresholds applied
afterwards, which matters because [occlusion](#occlusion) makes 10–20% of pixels
unmatchable on these scenes. On the score scale of [section 1.3](#13-the-score)
$$\lambda = \gamma = -0.1$$ reads as "reject anything worse than a tenth of the way
from chance to perfect".

#### One-to-one constraint

Also *uniqueness* or *mutual exclusivity*: at most one association per measurement
and at most one per object. It is physically true in stereo, because one surface
point projects once into each image. Enforcing it inside the inference is what this
whole series is testing, and [section 4.1](#41-what-uniqueness-is-worth) prices it
at +10.7 points of precision over [winner-take-all](#winner-take-all) on identical
scores.

#### Linear assignment problem

The combinatorial problem of choosing a maximum-weight one-to-one matching in a
bipartite graph — association with the uniqueness constraint and nothing else. It
is solvable exactly in polynomial time, which is why it is the right yardstick for
an approximate method. MASDA stops being a LAP the moment a factor is added that
couples two associations, such as the [ordering constraint](#ordering-constraint) —
that is the argument of [section 5](#5-speed-the-representation-decides-it).

#### Jonker-Volgenant and the Hungarian method

The exact LAP solvers. Kuhn's Hungarian method is the classical one; the
Jonker-Volgenant shortest-augmenting-path algorithm is the fast modern variant and
the one behind `scipy.optimize.linear_sum_assignment`. Used here as ground truth
for the *optimisation*, separately from ground truth for the *answer*:
[section 4.2](#42-against-the-exact-optimum) runs per-row JV over every row of two
scenes and finds MASDA's precision indistinguishable from exact.

> H. W. Kuhn (1955). *The Hungarian method for the assignment problem.* Naval
> Research Logistics Quarterly 2(1-2), 83-97.
> [doi:10.1002/nav.3800020109](https://doi.org/10.1002/nav.3800020109)

> R. Jonker, A. Volgenant (1987). *A shortest augmenting path algorithm for dense
> and sparse linear assignment problems.* Computing 38(4), 325-340.
> [doi:10.1007/BF02278710](https://doi.org/10.1007/BF02278710)

#### LP relaxation and the uniqueness condition

Relax the binary association variables to $$[0,1]$$ and the LAP becomes a linear
program. Bayati, Shah and Sharma proved max-product correct on bipartite matching
*provided that LP has a unique optimum* — and a tie between two candidate scores is
precisely how uniqueness fails. This is why the degenerate case in
[section 3.1](#31-reading-out-the-answer) is not a bug to patch: it is where the
guarantee stops. On an aggregated Census volume, which quantises to few levels,
exact ties are routine, so the condition fails on roughly half the rows
([section 4.2](#42-against-the-exact-optimum)) — measured here, not predicted by
the theory.

> M. Bayati, D. Shah, M. Sharma (2008). *Max-Product for Maximum Weight Matching:
> Convergence, Correctness, and LP Duality.* IEEE Transactions on Information
> Theory 54(3), 1241-1251.
> [doi:10.1109/TIT.2007.915695](https://doi.org/10.1109/TIT.2007.915695)

#### Objective ratio

The total score of MASDA's assignment divided by the exact optimum's, per row —
1.0 meaning "MASDA found an optimal assignment". Reported alongside precision in
[section 4.2](#42-against-the-exact-optimum) because the two disagree so
instructively: closing an 0.8% objective gap buys no precision at all, which makes
the objective a demonstrably loose proxy for the quantity a consumer experiences.

#### Greedy decode

The read-out rule: order candidates by [belief](#belief), accept a pair when both
sides agree and the score beats $$\lambda$$, then fill in what remains greedily by
belief. Not cosmetic — under near-ties every row's best belief points at the same
column, so mutual agreement alone commits one pair per row, and every greedily
accepted edge still raises the objective ([section 3.1](#31-reading-out-the-answer)).

#### Margin and the margin gate

The margin is best-minus-second-best in the same message currency: a per-pixel
confidence that MASDA produces as a by-product rather than as an extra pass. The
*margin gate* refuses to answer where the margin is small, trading coverage for
precision. It is the mechanism behind the 76% coverage in Part 2's SGM comparison,
and the reason a MASDA answer can be *absent* rather than merely wrong.

#### Segment reduction and max-excluding

Both message updates need "the maximum over this row except column $$j$$", which is
quadratic if computed per element and constant time if the largest and
second-largest are cached. On an edge list the same thing is a *segment reduction*
— a scatter-max over the edges grouped by endpoint (`np.maximum.at` in NumPy),
plus the count of elements attaining the maximum so ties are handled rather than
ignored. This is the whole trick that makes the sparse form $$O(E)$$;
[section 3.2](#32-the-sparse-matrix-form--this-is-the-design-not-an-optimisation)
gives the ten lines.

#### Ordering constraint

Along a rectified scanline, correspondences of an opaque surface preserve left-right
order: matches should not cross. Scanline dynamic programming gets this for free —
it is the classical argument for DP-based stereo, going back to Ohta and Kanade —
and a plain assignment formulation does not, which makes it the standing objection
to using MASDA here. [Section 6](#6-can-masda-express-the-ordering-constraint)
derives it as a pairwise factor costing one clamped scalar per conflicting edge,
verifies the closed form against brute-force max-sum, and then measures it as a
real but poor trade on the dense problem. Note that thin foreground objects
genuinely violate ordering, which is why $$\kappa$$ stays finite and why DP methods
need forbidden-move exceptions.

> Y. Ohta, T. Kanade (1985). *Stereo by Intra- and Inter-Scanline Search Using
> Dynamic Programming.* IEEE TPAMI 7(2), 139-154.
> [doi:10.1109/TPAMI.1985.4767639](https://doi.org/10.1109/TPAMI.1985.4767639)

#### Fenwick tree

A binary indexed tree: prefix sums and prefix counts in $$O(\log n)$$ with an array
and no pointers. It is the standard way to count crossing pairs without enumerating
them, and in [section 6.1](#61-measured-dense-it-works-and-it-costs-more-than-it-returns)
it turns from an optimisation into a precondition — with ~2400 crossing pairs per
row, the quadratic enumeration is the cost.

> P. M. Fenwick (1994). *A new data structure for cumulative frequency tables.*
> Software: Practice and Experience 24(3), 327-336.
> [doi:10.1002/spe.4380240306](https://doi.org/10.1002/spe.4380240306)

#### Sinkhorn, optimal transport and SuperGlue

The soft-assignment cousin of this formulation. Optimal transport asks for a
one-to-one-ish coupling between two distributions; adding an entropy term makes it
solvable by Sinkhorn's alternating row-and-column normalisation, and *dustbin* rows
and columns give unmatched items somewhere to go — which is $$\lambda$$ and
$$\gamma$$ under another name. Max-sum is the zero-temperature limit of that
relaxation. SuperGlue is the influential learned instance: same structure, but the
scores come from an attention network and the dustbin costs are learned rather than
hand-set, which is exactly the weakness [section 9](#9-what-would-improve-it)
admits to.

> M. Cuturi (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal
> Transport.* NeurIPS. [arXiv:1306.0895](https://arxiv.org/abs/1306.0895)

> P.-E. Sarlin, D. DeTone, T. Malisiewicz, A. Rabinovich (2020). *SuperGlue:
> Learning Feature Matching with Graph Neural Networks.* CVPR.
> [arXiv:1911.11763](https://arxiv.org/abs/1911.11763)

### A.3 Stereo vision

#### Rectification and epipolar geometry

Two cameras viewing one point define an *epipolar* geometry: the point's possible
positions in the second image lie on a line determined by the first image's
position. *Rectification* warps both images so those lines become the same
horizontal row in both, which collapses the correspondence search from
two-dimensional to one-dimensional and makes every image row an independent
problem — the property that makes this matcher embarrassingly parallel and, in
Part 3, a GPU kernel. Hartley and Zisserman's book is the standard treatment; Loop
and Zhang is the classic algorithm.

> C. Loop, Z. Zhang (1999). *Computing rectifying homographies for stereo vision.*
> CVPR. [doi:10.1109/CVPR.1999.786928](https://doi.org/10.1109/CVPR.1999.786928)

#### Disparity

The horizontal offset between a point's two projections, $$d = x_L - x_R$$, in
pixels. It is inverse depth: $$Z = fB/d$$ for focal length $$f$$ and baseline
$$B$$, so one disparity step is a large distance change far away and a small one up
close. $$D$$ throughout this series is the number of disparities searched (60 on
the camera, up to 80 on the benchmark scenes), and a *disparity plane* is the
whole image scored at one fixed $$d$$.

#### Occlusion

A surface visible in one camera and hidden in the other, so a correct match simply
does not exist. It affects 10–20% of pixels on these scenes, concentrated at depth
discontinuities — which is where stereo is hardest anyway — and it is the reason
[clutter and misdetection](#clutter-and-misdetection) have to be part of the model
rather than a threshold bolted on afterwards.

#### Cost volume

The $$W \times H \times D$$ array holding a matching score for every (pixel,
disparity) pair: 40 MB per frame at 450×375, 98 MB at 848×480. Building it,
aggregating it, then reading it back to pick winners is the textbook four-stage
pipeline codified in Scharstein and Szeliski's taxonomy. Part 2's second section is
about never materialising it — with two candidates per pixel, the running top-2
*is* the reduced volume — and Part 3 fuses away even the intermediate planes.

> D. Scharstein, R. Szeliski (2002). *A Taxonomy and Evaluation of Dense Two-Frame
> Stereo Correspondence Algorithms.* IJCV 47(1-3), 7-42.
> [doi:10.1023/A:1014573219977](https://doi.org/10.1023/A:1014573219977)

#### Census transform

A local binary descriptor: compare each neighbour in a window against the centre
pixel and keep one bit per comparison. A 7×7 window has 48 neighbours, so the
descriptor fits a `uint64`. Its value here is that it depends only on the *order*
of intensities, not their values, so it survives the gain and offset differences
between two real sensors — the property Hirschmüller and Scharstein measured across
matching costs under radiometric change. The *centre-symmetric* variant compares
opposing neighbour pairs instead, halving the bits; Part 2 measured it as worth 10%
runtime on the Jetson and 1.2 points of accuracy, and left it off.

> R. Zabih, J. Woodfill (1994). *Non-parametric local transforms for computing
> visual correspondence.* ECCV, LNCS 801, 151-158.
> [doi:10.1007/BFb0028345](https://doi.org/10.1007/BFb0028345)

> H. Hirschmüller, D. Scharstein (2009). *Evaluation of Stereo Matching Costs on
> Images with Radiometric Differences.* IEEE TPAMI 31(9), 1582-1599.
> [doi:10.1109/TPAMI.2008.221](https://doi.org/10.1109/TPAMI.2008.221)

#### Hamming distance and popcount

The distance between two Census descriptors is the number of differing bits: XOR
them and count the ones. `popcount` is that count in hardware —
`__builtin_popcountll` becomes one instruction on x86-64 and a short sequence
around NEON's `cnt` on ARMv8 — which is what makes a 48-bit descriptor comparison
essentially free. Two unrelated descriptors agree on half
their bits by chance, which is the zero point the score scale of
[section 1.3](#13-the-score) is built around.

#### Truncated absolute difference

The absolute intensity difference between two pixels, clipped at a maximum. On its
own it is a weak matching cost; added to Census it contributes a *graded* signal
where Census is nearly saturated, which Part 2 measured at 10.3% → 9.7% bad-1.0 by
itself. Combining a binary descriptor with an intensity term this way is
established practice — ADCensus is the well-known instance.

> X. Mei, X. Sun, M. Zhou, S. Jiao, H. Wang, X. Zhang (2011). *On building an
> accurate stereo matching system on graphics hardware.* ICCV Workshops.
> [doi:10.1109/ICCVW.2011.6130280](https://doi.org/10.1109/ICCVW.2011.6130280)

#### Cost aggregation

Summing or filtering each candidate's score over a neighbourhood, so the decision
rests on a region's evidence rather than one pixel's. It is the third stage of the
classical taxonomy and, on this project, the single largest accuracy lever: one
7×7 Census comparison quantises to 49 levels, and Part 2 measured unaggregated
scores at 28.1% bad against 12.7% for SGM stripped of its smoothness term
entirely. In factor-graph language, better unaries beat a cheap pairwise term —
which is also why the cheap smoothness factor is a recorded negative result.

#### Edge-aware recursive filter

The aggregation actually used: a two-pass recurrence per axis whose per-pixel
coefficient shrinks where the image gradient is large, so support stops at
intensity edges instead of blurring across depth boundaries. Its cost is $$O(1)$$
per pixel independently of support size, which is why the support can be large. The
family it belongs to is the domain-transform and non-local aggregation line of
work; the exact recurrence, the integer coefficients and the truncation matter here
because Part 3 reproduces them bit-for-bit on the GPU.

> E. S. L. Gastal, M. M. Oliveira (2011). *Domain transform for edge-aware image
> and video processing.* ACM TOG 30(4).
> [doi:10.1145/2010324.1964964](https://doi.org/10.1145/2010324.1964964)

> Q. Yang (2012). *A non-local cost aggregation method for stereo matching.* CVPR.
> [doi:10.1109/CVPR.2012.6247827](https://doi.org/10.1109/CVPR.2012.6247827)

#### Smoothness prior

A pairwise term that penalises disparity differences between neighbouring pixels,
on the argument that surfaces are mostly continuous. It is the mechanism
[SGM](#semi-global-matching) is built on, and the one this factor graph does *not*
have. Two cheap versions were tried here and are recorded as negatives — worse at
every weight — with the mechanism: on a quantised Census volume the information a
neighbour smoothness term would add is better added by
[aggregating the unaries](#cost-aggregation). The interesting unbuilt object is a
graph carrying both uniqueness and path smoothness at once
([section 9](#9-what-would-improve-it)).

#### Keypoints and detector repeatability

A *keypoint* is a distinctive image location found by a detector (FAST, Harris,
ORB) and summarised by a descriptor, so matching considers only a few hundred
points per image instead of every pixel. *Repeatability* is the fraction of
keypoints found in one image that the detector also finds in the other — and it
bounds recall before any matcher runs, since a point detected in only one view
cannot be matched. This series began as a sparse-keypoint study and moved to the
dense problem when that bound was measured at under 51% on this camera: the dense
formulation deletes the detector, and with it the ceiling. Parts 1 and 2 refer back
to "the original keypoint study" in that sense.

#### Winner-take-all

Take each pixel's best-scoring disparity and stop — no uniqueness, no interaction
between pixels. It is the baseline every stereo matcher has to beat, and the
comparison that isolates what the [one-to-one constraint](#one-to-one-constraint)
contributes, because it can be run on *identical* scores
([section 4.1](#41-what-uniqueness-is-worth)): 0.776 precision against MASDA's
0.884.

#### Semi-global matching

SGM: the standard fast dense matcher. It aggregates a smoothness penalty along
several one-dimensional paths across the image and sums the results, approximating
a 2-D energy at 1-D cost. It has **no uniqueness constraint** — a
[left-right consistency check](#left-right-consistency-check) is added afterwards,
which costs a second matcher run — and no native confidence. It is the direct
competitor here, measured in Part 2 at 10.9% bad-1.0 against dense MASDA's 9.7%.
The two mechanisms are orthogonal, and a factor graph carrying both uniqueness and
path smoothness is the interesting object neither tool implements today.

> H. Hirschmüller (2008). *Stereo Processing by Semiglobal Matching and Mutual
> Information.* IEEE TPAMI 30(2), 328-341.
> [doi:10.1109/TPAMI.2007.1166](https://doi.org/10.1109/TPAMI.2007.1166)

#### Left-right consistency check

Match left-to-right, match right-to-left, and keep only the pixels where the two
agree. The standard way to detect occlusions and mismatches after the fact, and the
standard way to get uniqueness without modelling it — at the price of running the
matcher twice. MASDA gets the same property inside one inference pass.

> P. Fua (1993). *A parallel stereo algorithm that produces dense depth maps and
> preserves image features.* Machine Vision and Applications 6(1), 35-49.
> [doi:10.1007/BF01212430](https://doi.org/10.1007/BF01212430)

#### Sub-pixel disparity

Fitting a curve to the scores around the winning integer disparity to recover a
fractional answer. It is the refinement stage of the classical taxonomy and it
matters more for metric depth accuracy than anything else downstream — and it is
not implemented here, which [section 9](#9-what-would-improve-it) records as an
open item rather than a detail.

#### Coarse-to-fine

Solve at half resolution, upsample the answer, and search only a narrow band around
it at full resolution — a pyramid strategy, and the family every fast published CPU
matcher belongs to in some form. Part 2 measured the ceiling honestly (81.5% of
known pixels keep the truth inside a ±2 band) and then measured the delivery: 5.2×
less arithmetic, 1.2–1.4× faster on the desktop, and *flat* on the Jetson, for
three specific reasons that are the most useful negative result in the series.

#### Repetitive texture and ambiguity

A periodic pattern produces several near-equal scores one period apart, so the
evidence genuinely does not identify the answer. This is the failure mode that
bounds every method on these scenes including the exact solver — uniqueness can
rule out a *conflict*, but a patch matched one period off conflicts with nothing.
It is also why the gain from uniqueness is largest exactly where the scene is worst
([section 4.1](#41-what-uniqueness-is-worth)).

#### Disparity-major and disparity-minor layout

Two ways to index the cost volume: `[d][x]` keeps whole constant-disparity planes
contiguous, `[x][d]` keeps each pixel's run of disparities contiguous. The choice
is not a style question but a memory-system question, and Parts 2 and 3 answer it
in *opposite* directions — the CPU wants planes resident in L2, the GPU wants the
innermost index to be the one a warp spans. Part 3's section 4 is the resolution:
neither layout is wrong; trying to make one implementation serve both was.

### A.4 Datasets and metrics

#### Middlebury stereo datasets

The standard rectified stereo benchmark with dense ground truth. This series uses
Teddy and Cones from the 2003 set and the six 2005 scenes (Art, Books, Dolls,
Laundry, Moebius, Reindeer) at third size, whose [dataset
page](https://vision.middlebury.edu/stereo/data/) grants permission to use and
publish the images and disparity maps. The images are not redistributed in this
repository; [section 2](#2-ground-truth) also documents the disparity scale factor
of 3 being *established* by an identity check rather than assumed, because the 2005
archives do not document it.

> D. Scharstein, R. Szeliski (2003). *High-accuracy stereo depth maps using
> structured light.* CVPR, 195-202.
> [doi:10.1109/CVPR.2003.1211354](https://doi.org/10.1109/CVPR.2003.1211354)

> D. Scharstein, C. Pal (2007). *Learning conditional random fields for stereo.*
> CVPR. [doi:10.1109/CVPR.2007.383191](https://doi.org/10.1109/CVPR.2007.383191)

#### Structured-light ground truth

How that ground truth was made: project coded light patterns onto the scene so
every surface point identifies itself, and the correspondence becomes
unambiguous — accurate to about a quarter pixel here, with a few percent of pixels
left unknown. Those unknowns are marked as disparity zero rather than by a separate
mask, which is why [section 2](#2-ground-truth) excludes them from precision
instead of scoring them as wrong.

#### Coverage, precision and the bad pixel rate

Three numbers that have to be read together. *Coverage* is the fraction of pixels
the matcher answers at all — a matcher with a [margin gate](#margin-and-the-margin-gate)
answers fewer. *Precision* is the fraction of answered pixels with known ground
truth that are correct within a tolerance (1 px throughout). *bad-1.0* is the
complement, the percentage wrong by more than 1 px, and it is the metric Part 2
reports because it is what the stereo literature reports. The product,
coverage × (1 − bad), is *correct-over-known*: the share of all knowable pixels
actually delivered correctly. Comparing two matchers on one of the three alone is
how a matcher that mostly abstains looks excellent.

#### Interleaved best-of-N

The timing protocol on the Jetson: run the variants alternately within one session
and take each one's minimum, rather than running all of A then all of B. The board's
run-to-run variance is 37% at locked clocks and stable temperature, because six
heterogeneous cores get scheduled differently each run, so a single measurement is
worthless and consecutive blocks let drift masquerade as an effect.

### A.5 The matchers and methods this one is measured against

#### ELAS

Efficient Large-Scale Stereo Matching: match a sparse set of robust support points
first, triangulate them into a prior, and search only near the resulting surface.
The canonical "don't sweep the whole disparity range" CPU matcher, and the direct
ancestor of the [coarse-to-fine](#coarse-to-fine) experiment Part 2 measures.

> A. Geiger, M. Roser, R. Urtasun (2011). *Efficient Large-Scale Stereo Matching.*
> ACCV 2010, LNCS 6492, 25-38.
> [doi:10.1007/978-3-642-19315-6_3](https://doi.org/10.1007/978-3-642-19315-6_3)

#### PatchMatch stereo

Avoids the sweep from the other direction: instead of testing every disparity,
propagate good hypotheses — including slanted support planes — between neighbouring
pixels and refine them randomly.

> M. Bleyer, C. Rhemann, C. Rother (2011). *PatchMatch Stereo — Stereo Matching
> with Slanted Support Windows.* BMVC.
> [doi:10.5244/C.25.14](https://doi.org/10.5244/C.25.14)

#### rSGM

Semi-global matching engineered for the CPU: SIMD throughout, fewer paths, and
subsampled far disparities, reaching VGA at 128 disparities above 16 Hz on a
desktop CPU. The reference point for "how fast should a careful CPU stereo matcher
be".

> M. Spangenberg, T. Langner, S. Adfeldt, R. Rojas (2014). *Large scale
> Semi-Global Matching on the CPU.* IEEE Intelligent Vehicles Symposium.
> [doi:10.1109/IVS.2014.6856419](https://doi.org/10.1109/IVS.2014.6856419)

#### ReS2tAC

Real-time SGM on exactly the hardware class this project targets — embedded ARM
with NEON, and CUDA on Jetson-class GPUs. Two of its design decisions were
re-derived independently here before being recognised: putting *disparity* in the
SIMD lanes rather than pixels (Part 2's NEON kernel) and the same assignment for
CUDA warps (Part 3's layout). It is the closest published prior art to this
pipeline's engineering, on both processors.

> B. Ruf, J. Mohrs, M. Weinmann, S. Hinz, J. Beyerer (2021). *ReS2tAC — UAV-Borne
> Real-Time SGM Stereo Optimized for Embedded ARM and CUDA Devices.* Sensors
> 21(11), 3938. [doi:10.3390/s21113938](https://doi.org/10.3390/s21113938)

#### ESPReSSo

Slanted PatchMatch made real-time for spacetime stereo, with edge-aware
aggregation under shared plane hypotheses — which is why Part 2 lists it among the
work behind this pipeline's aggregation stage.

> H. Nover, S. Achar, D. B. Goldman (2018). *ESPReSSo: Efficient Slanted
> PatchMatch for Real-Time Spacetime Stereo.* 3DV.
> [doi:10.1109/3DV.2018.00072](https://doi.org/10.1109/3DV.2018.00072)

#### GPU-efficient recursive filtering

The standard way to parallelise a recurrence on a GPU: algebraically reassociate it
into blocks so the blocks can run concurrently. Part 3 could not use it — the
reassociation does not reproduce the CPU's integer truncation bit-for-bit, and
bit-identity was the referee — and the entry is here because that paper's existence
is also the explanation of why the warp-serial alternative loses.

> D. Nehab, A. Maximo, R. S. Lima, H. Hoppe (2011). *GPU-Efficient Recursive
> Filtering and Summed-Area Tables.* ACM TOG 30(6), SIGGRAPH Asia.
> [doi:10.1145/2024156.2024210](https://doi.org/10.1145/2024156.2024210)

### A.6 Hardware and implementation

#### Jetson TX2

The target board: an NVIDIA Tegra X2 module with six ARM cores, a 256-core Pascal
GPU, and one LPDDR4 memory system shared between them. Everything in Parts 2 and 3
that is surprising traces back to that last clause — the CPU and the GPU compete
for the same bandwidth, and the board's memory streams are its weakest point
relative to a desktop. [NVIDIA's developer
page](https://developer.nvidia.com/embedded/jetson-tx2) is the specification.

#### A57 and Denver cores

The six cores are not identical: four ARM Cortex-A57s and two of NVIDIA's own
Denver cores, a wider design with dynamic code optimisation. Heterogeneity is why
run-to-run variance is 37% ([interleaved best-of-N](#interleaved-best-of-n)) and
why thread pinning has *opposite* signs in the two tools — Part 3 measures pinning
the solve to the A57 cluster as essential for the GPU pipeline and 40% worse for the
CPU-only matcher, which needs the Denvers' throughput.

#### RealSense D435 and the IR pair

The camera this project is built around: an Intel RealSense D435, used not for its
own depth output but for its two infrared cameras as a raw rectified stereo pair at
848×480 — which is where that resolution, and the 33.3 ms frame budget, come from.
[Product page](https://www.intelrealsense.com/depth-camera-d435/).

#### NEON and SIMD

Single Instruction Multiple Data: one instruction operating on several values at
once. NEON is ARM's 128-bit version — eight `int16` lanes, against AVX2's 256 bits
on the desktop, which is why halving the data type was worth 20% on the Jetson and
nothing on the desktop. `vqtbl4q_u8` is the NEON table-lookup instruction across
four registers that made the score loop's two lookup tables vector operations.
[ARM's intrinsics reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
is searchable per instruction.

#### Q14 fixed point

Integer arithmetic standing in for fractions: an `int16` holding a value scaled by
$$2^{14}$$, so $$[-2, 2)$$ is representable to about $$6 \times 10^{-5}$$. The
score is naturally in $$[-1, 1]$$, so it fits with headroom to spare, and the whole
pipeline stays integer — which is what makes bit-identity between CPU and GPU
achievable at all, since integer arithmetic has no reassociation freedom.

#### Warp, coalescing and shuffle

A CUDA *warp* is 32 threads executing together. *Coalescing* is the hardware
merging their memory accesses into as few aligned transactions as possible — 32
consecutive addresses become one 128-byte transaction, while a strided pattern
becomes 32 of them, which is the entire reason Part 3's volume is indexed
disparity-minor. A *shuffle* (`__shfl_down_sync`) exchanges registers directly
between lanes of a warp, and is how the top-2 reduction works — including the
tie-break bug in Part 3's section 6, which follows from the shuffle tree merging
non-adjacent ranges. The
[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
is the reference.

#### Constant and shared memory

Two of the GPU's addressable memories. `__constant__` is optimised for *broadcast*
— all lanes reading one address — and serializes into replays when lanes read
different addresses, which is what 32 different Hamming distances per warp do by
construction. Shared memory is per-block scratch with no such penalty, and moving
the lookup tables there was worth ~2 ms in Part 3.

#### Pinned memory and I/O coherency

*Pinned* (page-locked) host memory is what a GPU DMA engine can copy without the
CPU's involvement, and on discrete GPUs it is also fast for the CPU to read. On the
TX2 it is not: the SoC has no I/O coherency, so every `cudaHostAlloc` flavour —
including the one whose name suggests otherwise — is uncached from the CPU's side,
and a solver reading candidates out of it runs about seven times slower than from
ordinary pageable memory. The fix is a staged copy into a plain `std::vector`, and
NVIDIA's [CUDA for Tegra
notes](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html) document the
memory-coherency model this violates.

#### Kernel fusion

Merging two passes so the intermediate never reaches memory. The principle Part 3
states — *a pass that stores exactly what the next pass reads is a fusion
candidate, and a store nothing reads afterwards is a bug you are paying for* —
found time three times on the GPU and *nothing* on the CPU, where the intermediate
plane was already L2-resident. Same transformation, opposite verdict — the series'
thesis in one experiment.

#### Bandwidth bound

A kernel is bandwidth bound when its runtime is set by bytes moved rather than
arithmetic performed. Dividing each kernel's bytes by its measured time and
comparing against what the board actually achieves (~35–40 GB/s here) turns
optimisation from guesswork into arithmetic: a kernel at 19 GB/s is leaving half
the machine idle, and it also says when to *stop*. The roofline model is the formal
version of this reasoning.

> S. Williams, A. Waterman, D. Patterson (2009). *Roofline: an insightful visual
> performance model for multicore architectures.* Communications of the ACM 52(4),
> 65-76. [doi:10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785)

#### Amdahl's law

The speedup available from parallelising a program is capped by the part that stays
serial. Quoted here because it was the answer after five wrong guesses: the workers
were 5.99 of 6 cores busy while alive, and the missing time was a serial prologue
and a serial-ish merge *outside* the pool. The instrument that showed it — per-thread
span printed next to per-thread busy — is the transferable part.

> G. M. Amdahl (1967). *Validity of the single processor approach to achieving
> large scale computing capabilities.* AFIPS Spring Joint Computer Conference.
> [doi:10.1145/1465482.1465560](https://doi.org/10.1145/1465482.1465560)

#### Autovectorisation

The compiler turning a scalar loop into SIMD instructions by itself. It is worth
knowing about mainly because it is easy to destroy: putting a branchy insert inside
the recurrence loop cost 18% outright in Part 3's section 8. On a CPU, fuse loops
only if the hot loop stays branch-free.

#### Bit-identity

Requiring two implementations to produce byte-identical output, checked with `cmp`.
Part 3 treats it as a design constraint rather than a test — every GPU intermediate
replicates the CPU's exact integer arithmetic — and it paid twice: a ten-pixel
tie-break error in 407,040 and a padding bug that silently broke six of eight
scenes are both invisible to an accuracy benchmark at the scale they occur. The
caveat is real too: `cmp` between two *multi-threaded* runs is not an identity
check, because dynamic work distribution changes which of two equal scores wins.
