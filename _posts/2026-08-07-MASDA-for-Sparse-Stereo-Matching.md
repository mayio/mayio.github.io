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
I derived [MASDA](/masda-glossary/#masda) for the tracking problem: associate measurements with
objects, allowing for [clutter and misdetection](/masda-glossary/#clutter-and-misdetection). Stereo
matching has the same structure, so this post applies MASDA there — to the *dense*
problem, a [disparity](/masda-glossary/#disparity) for every pixel — and measures what it gains.

The formulation runs on **sparse matrices**. Each pixel offers only its two best
disparity candidates out of the aggregated [cost volume](/masda-glossary/#cost-volume), so the
association graph carries two edges per pixel instead of a $$W \times W$$ matrix
per row, and every
result below depends on that representation twice over: it is what makes the exact
comparison feasible, and it is what makes the solver fast.

*[Part 2](https://www.mariolueder.com/2026-08-08-Dense-MASDA-Belief-Propagation-Stereo-on-a-Jetson/)
takes this formulation into C++ on a Jetson TX2 and measures it against SGM;
[Part 3](https://www.mariolueder.com/2026-08-09-Realtime-Dense-MASDA-on-the-Jetson-GPU/)
makes it real-time on the TX2's GPU, bit-identically.*

*Every technical term this series uses — from factor graphs to CUDA warps — is
defined in the [glossary for this series](/masda-glossary/), with links to the
original work. Terms link there on first use, in all three parts.*

Results, briefly — pooled over eight [Middlebury](/masda-glossary/#middlebury-stereo-datasets)
scenes with [structured-light ground truth](/masda-glossary/#structured-light-ground-truth), roughly
1.3 million answers:

- **The [one-to-one constraint](/masda-glossary/#one-to-one-constraint) is worth +10.7 points of
  [precision](/masda-glossary/#coverage-precision-and-the-bad-pixel-rate) over
  [winner-take-all](/masda-glossary/#winner-take-all) on identical scores**: 0.884 against 0.776,
  while keeping 96% of WTA's correct answers. Per scene the gain is 8–13 points,
  largest where the texture is worst.
- **Loopy [max-sum](/masda-glossary/#max-sum-max-product-and-sum-product) matches the exact
  [assignment](/masda-glossary/#linear-assignment-problem) optimum on precision** where the
  exact optimum is computable: per-row
  [Jonker-Volgenant](/masda-glossary/#jonker-volgenant-and-the-hungarian-method) reaches 0.914 and
  0.941 on Teddy and Cones; MASDA reaches 0.915 and 0.942.
- It does that from **two message-passing iterations**, sitting 0.8% short of the
  optimal [objective](/masda-glossary/#objective-ratio) with **1.6% of rows exactly optimal**.
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

In tracking, [data association](/masda-glossary/#data-association) puts measurements
$$i \in \{1 \dots m\}$$ with objects $$j \in \{1 \dots n\}$$, at most one each way,
with the option of calling a measurement clutter or an object misdetected.

For stereo, substitute the nouns — per
[rectified](/masda-glossary/#rectification-and-epipolar-geometry) image row. Pixels in the left
row are measurements, pixels in the right row are objects. At most one
association each way, because one surface point produces one projection per
image. A left pixel whose surface is occluded in the right view has no partner
at all, which is clutter ($$\lambda$$). A right pixel whose surface is hidden
from the left camera is a misdetection ($$\gamma$$). [Occlusion](/masda-glossary/#occlusion) is not
a small
correction: on these scenes it affects 10–20% of pixels, and a formulation that
assumes every pixel is matchable is wrong exactly at the depth discontinuities
where stereo is hardest.

What stereo adds is geometry. The pair is rectified, so correspondences lie on
the same image row and disparity $$d = x_L - x_R$$ is positive and bounded. Each
row is an independent association problem — which makes the whole thing
parallel, and later, a GPU workload.

### 1.1 Factor graph

Same as the tracking case, on the same [factor graph](/masda-glossary/#factor-graph): binary
association variables $$c_{ij}$$, clutter
indicators $$e_i$$, misdetection indicators $$\delta_j$$, similarity factors $$S_{ij}$$,
clutter factors $$\Lambda_i$$, misdetection factors $$\Gamma_j$$, and the exclusivity
constraints $$I_i$$ and $$E_j$$.

The graph is loopy. Every $$c_{ij}$$ sits in both an $$I_i$$ and an $$E_j$$ constraint, so
there are four-cycles everywhere. Hence [loopy belief
propagation](/masda-glossary/#loopy-belief-propagation), and hence no
[convergence guarantee](/masda-glossary/#convergence-and-the-anytime-property).

### 1.2 Messages

The [two message directions](/masda-glossary/#messages-responsibility-and-availability) are
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

[Damping](/masda-glossary/#damping) on both:

$$
x^{(t+1)} \leftarrow (1-\eta)\, x_{\text{target}} + \eta\, x^{(t)}
$$

The [belief](/masda-glossary/#belief) combines both directions,

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

Descriptors are the [Census transform](/masda-glossary/#census-transform) over a
$$7 \times 7$$ window: one bit per
neighbour, set when the neighbour is darker than the centre. 48 bits fit a
`uint64`, so the distance is a single
[`popcount`](/masda-glossary/#hamming-distance-and-popcount), and Census is invariant to
monotonic intensity mappings, which absorbs gain and offset differences between
two real sensors.

Two unrelated Census descriptors agree on half their bits by chance, so scaling
the [Hamming distance](/masda-glossary/#hamming-distance-and-popcount) $$h$$ around that point
gives a score with a usable zero:
$$+1$$ perfect, $$0$$ chance. On this scale $$\lambda = \gamma = -0.1$$ means
"reject anything worse than a tenth of the way from chance to perfect", which is
easier to reason about than a tuned constant.

One pixel's Census comparison quantises to only 49 levels, which is not enough
signal per pixel. The score MASDA actually consumes is
[*aggregated*](/masda-glossary/#cost-aggregation) over an
edge-aware support region — an [O(N) recursive
filter](/masda-glossary/#edge-aware-recursive-filter) that stops at intensity
edges — so each candidate's score summarises a neighbourhood while respecting
depth boundaries. The aggregation machinery, and the measurements behind each of
its choices, are Part 2's subject; here it is the given: the sparse candidate
matrix is built from the aggregated volume of the shipping implementation, so
this study and the production pipeline score identical evidence.

---

## 2. Ground truth

The matcher is measured on **Teddy** and **Cones** from the [Middlebury
2003 stereo set](/masda-glossary/#middlebury-stereo-datasets), and on the six Middlebury 2005
scenes: Art, Books, Dolls, Laundry,
Moebius and Reindeer, at third size. They are rectified, and they ship
[structured-light ground truth](/masda-glossary/#structured-light-ground-truth) at quarter-pixel
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
optimum](/masda-glossary/#lp-relaxation-and-the-uniqueness-condition), and uniqueness is exactly
what Bayati, Shah and
Sharma require for [max-product](/masda-glossary/#max-sum-max-product-and-sum-product) to be
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

So: [order by belief, decide by $$\lambda$$](/masda-glossary/#greedy-decode), require row and column
to agree, then fill
in greedily by belief over what is left. The greedy fill is not cosmetic: under
near-ties every row's best belief points at the same column, so requiring mutual
agreement commits exactly one pair, and every greedily accepted edge has
$$s > \lambda$$ and two free endpoints, so it raises the objective.

### 3.2 The sparse-matrix form — this is the design, not an optimisation

Put the messages on the edges. The only awkward part is that both updates need
$$\max_{k \neq j}$$ over a row or column, which is quadratic in row length if done
directly. Three [segment reductions](/masda-glossary/#segment-reduction-and-max-excluding) answer it
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

A note on [damping](/masda-glossary/#damping): undamped max-sum on heavily tied problems does not
settle; the
largest message change plateaus instead of decaying. Damping of 0.3–0.5 stabilises
it, and solution quality is flat across that range. [Message convergence is not the
property you need](/masda-glossary/#convergence-and-the-anytime-property) anyway — the *decision*
stabilises long before the messages do,
and [section 4.3](#43-two-iterations-against-thirty) measures how much sooner:
fifteen times sooner, on these scenes.

---

## 4. Results against ground truth

$$\lambda = \gamma = -0.1$$, **two iterations** — the shipping configuration —
damping 0.4, candidates = top-2 per pixel from the aggregated volume, tolerance
1 px, unknown ground truth excluded. ([Section 4.3](#43-two-iterations-against-thirty)
measures what the other 28 iterations would buy, which is nothing.)
Three solvers on **identical scores**: [winner-take-all](/masda-glossary/#winner-take-all) (argmax
over the full volume — no uniqueness), MASDA on the sparse candidate matrix, and
per-row exact
assignment ([Jonker-Volgenant](/masda-glossary/#jonker-volgenant-and-the-hungarian-method) with
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
[keypoint study's](/masda-glossary/#keypoints-and-detector-repeatability) central finding — the
constraint pays in proportion to the [ambiguity](/masda-glossary/#repetitive-texture-and-ambiguity) —
now measured on three orders of magnitude more answers.

For the full engineered pipeline (the C++ implementation with its
[margin gate](/masda-glossary/#margin-and-the-margin-gate),
which trades a little coverage for precision), the comparison against OpenCV's
[SGM](/masda-glossary/#semi-global-matching) lands at
**9.7% [bad-1.0](/masda-glossary/#coverage-precision-and-the-bad-pixel-rate) against SGM's 10.9%**
at 76% versus 78% [coverage](/masda-glossary/#coverage-precision-and-the-bad-pixel-rate) —
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
MASDA's [objective ratio](/masda-glossary/#objective-ratio) against JV was 1.0000 — it simply *was*
optimal, three
problems out of three. Here it never is: 0.9918 at the shipping setting, and even
at thirty iterations it reaches the exact optimum on only 47% and 69% of rows. The
difference is ties. An aggregated Census volume quantises to few enough levels
that exactly tied candidates are everywhere, so the
[LP-uniqueness condition](/masda-glossary/#lp-relaxation-and-the-uniqueness-condition) of the
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
need](/masda-glossary/#convergence-and-the-anytime-property).** The
messages are still moving at iteration thirty; the decision stopped moving around
iteration two. Every number elsewhere in this article is therefore reported at the
shipping setting, and I would treat any belief-propagation matcher quoting an
iteration count without this comparison with suspicion — including my own earlier
version of this page.

---

## 5. Speed: the representation decides it

The complexity argument is $$O(T \cdot E)$$ against
[Jonker-Volgenant's](/masda-glossary/#jonker-volgenant-and-the-hungarian-method)
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
is [anytime](/masda-glossary/#convergence-and-the-anytime-property), incremental, and accepts
factors that destroy the assignment
structure, where a [LAP solver](/masda-glossary/#linear-assignment-problem) cannot follow.

---

## 6. Can MASDA express the ordering constraint?

Scanline stereo methods use the [ordering constraint](/masda-glossary/#ordering-constraint): matches
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
\log E_r)$$ [Fenwick-tree](/masda-glossary/#fenwick-tree) construction I waved at in the keypoint
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
[repetitive texture](/masda-glossary/#repetitive-texture-and-ambiguity) matched one period off
crosses nothing at all.

## 7. Comparison with existing work

**[Jonker-Volgenant / Hungarian](/masda-glossary/#jonker-volgenant-and-the-hungarian-method).**
Exact, $$O(N^3)$$, and here exactly as good as
MASDA where it can be run at all ([section 4.2](#42-against-the-exact-optimum)).
For a pure assignment problem of moderate size, use it. MASDA earns its place on
speed at scale — 6.4× in NumPy, far more engineered — and when you intend to add
factors that stop the problem being a LAP.

> Jonker, R., & Volgenant, A. (1987). *A shortest augmenting path algorithm for
> dense and sparse linear assignment problems.* Computing, 38(4), 325-340.
> [doi:10.1007/BF02278710](https://doi.org/10.1007/BF02278710)

**[SPADA / sum-product data association](/masda-glossary/#spada).** Produces marginal association
probabilities rather than a MAP assignment, at higher cost. If the consumer wants
soft weights, that is the right choice. Stereo wants a decision per pixel, so MAP
is what is needed.

**[Sinkhorn and optimal transport, as in
SuperGlue](/masda-glossary/#sinkhorn-optimal-transport-and-superglue).** Structurally very close: soft
one-to-one assignment with dustbins, which are $$\lambda$$ and $$\gamma$$ under another
name. Sinkhorn is the entropy-regularised relaxation and max-sum is its
zero-temperature limit. SuperGlue's real advantage is that its scores come from a
learned attention network instead of a hand-designed $$s(i,j)$$, and its dustbin costs
are learned rather than set. That points straight at the weakest part of what I have
here.

> Sarlin, P.-E., DeTone, D., Malisiewicz, T., & Rabinovich, A. (2020).
> *SuperGlue: Learning Feature Matching with Graph Neural Networks.* CVPR.
> [arXiv:1911.11763](https://arxiv.org/abs/1911.11763)

**[Semi-global matching](/masda-glossary/#semi-global-matching).** The standard fast dense method,
and the direct
competitor now that this formulation is dense too. SGM aggregates
[smoothness](/masda-glossary/#smoothness-prior)
along scanline paths and has **no uniqueness constraint at all** — a
[left-right consistency check](/masda-glossary/#left-right-consistency-check) is bolted on
afterwards, which costs a second matcher run.
MASDA gets mutual exclusivity inside the inference, in one run, plus a per-pixel
confidence (the [margin](/masda-glossary/#margin-and-the-margin-gate)) as a by-product. Measured head-to-head in Part 2:
**9.7% bad-1.0 against SGM's 10.9%** over these eight scenes, at 76% against 78%
coverage. The two mechanisms are orthogonal, and the interesting object — a
factor graph with both uniqueness and path smoothness — does not exist in either
tool today.

> Hirschmüller, H. (2008). *Stereo Processing by Semiglobal Matching and Mutual
> Information.* IEEE TPAMI, 30(2), 328-341.
> [doi:10.1109/TPAMI.2007.1166](https://doi.org/10.1109/TPAMI.2007.1166)

**[ELAS](/masda-glossary/#elas)** narrows a dense search around triangulated support points — the
avoid-the-sweep family, alongside [PatchMatch](/masda-glossary/#patchmatch-stereo) and
[rSGM](/masda-glossary/#rsgm). Part 2 measured this project's version of that idea (a
[coarse-to-fine](/masda-glossary/#coarse-to-fine) mask) at accuracy parity and recorded exactly
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
  [LAP solver](/masda-glossary/#linear-assignment-problem).
  Adding an [ordering](/masda-glossary/#ordering-constraint), [smoothness](/masda-glossary/#smoothness-prior) or
  temporal factor keeps a [factor graph](/masda-glossary/#factor-graph) a factor
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

**A [smoothness factor](/masda-glossary/#smoothness-prior), done properly.** (Ordering is now
measured rather than
open — section 6 — and the interesting question it leaves is whether a
neighbourhood factor pays where ordering did not.) Neighbouring pixels on the same surface
have similar disparity, and the current factor graph ignores it. The cheap
variants are measured negatives (Part 2's record); the real derivation — path
aggregation as factors, so uniqueness and smoothness live in one graph — is the
interesting object this formulation makes possible.

**[Sub-pixel disparity](/masda-glossary/#sub-pixel-disparity).** The candidates are integer; the fit
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

Every term this article uses — factor graphs, max-sum messages, Census, cost
aggregation, SGM, the Jetson's memory system — is defined in the
[glossary for this series](/masda-glossary/), with links to the original work
behind each one. Terms link there on first use, in all three parts.
