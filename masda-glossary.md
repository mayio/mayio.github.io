---
layout: page
title: "Glossary: belief propagation, data association and stereo matching"
subtitle: "Every term the MASDA dense-stereo series uses — factor graphs, max-sum messages, Census, cost aggregation, SGM, CUDA warps, Jetson TX2 memory — defined once, with links to the original work."
permalink: /masda-glossary/
mathjax: true
---

The MASDA stereo series moves between three fields — inference on graphical
models, stereo vision, and embedded performance work — and each brings its own
vocabulary. Every term the three parts use is defined here once, with a link to
the work it comes from, so that no part assumes the others have been read.

The series:

1. [Dense Stereo Matching with Max-Sum Belief Propagation on Sparse Matrices
   (MASDA)](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/) — the formulation, and what the one-to-one constraint is worth
   against ground truth.
2. [Dense Stereo Matching with Max-Sum Belief Propagation on a Jetson TX2](https://www.mariolueder.com/2026-08-08-Dense-MASDA-Belief-Propagation-Stereo-on-a-Jetson/)
   — the same algorithm as a shipping C++ matcher, measured against SGM.
3. [Real-Time Dense Stereo at 34 Hz on a Jetson TX2](https://www.mariolueder.com/2026-08-09-Realtime-Dense-MASDA-on-the-Jetson-GPU/) — the image plane on
   the GPU, the graph on the CPU, bit-identically.

Terms in all three link here on first use. If the inference machinery is new to
you, the [Loopy Belief Propagation series](/belief-propagation/) builds it from
first principles with the Python to run.

One convention worth stating: where an entry states something as *measured on
this project* rather than as established practice, it says so and names the
section that measured it. The distinction between "this is what the field knows"
and "this is what my board did" is the one most worth keeping visible.

## Inference on factor graphs

### Factor graph

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

### Loopy belief propagation

Belief propagation is exact on a factor graph that is a tree: one sweep in each
direction and every node knows its answer. Run the same local update rule on a
graph *with cycles* and you get loopy belief propagation — no guarantee of
convergence and no guarantee of correctness, but in practice often excellent. The
association graph here is loopy by construction: every association variable sits in
both a row constraint and a column constraint, so four-cycles are everywhere
([section 1.1](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#11-factor-graph)). Weiss and Freeman give the theoretical account
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

### Max-sum, max-product and sum-product

Three names for the same message-passing skeleton with different operators inside
it. *Sum-product* marginalises: it computes, for each variable, the probability of
each of its states. *Max-product* maximises instead of summing, so it seeks the
single best joint configuration. *Max-sum* is max-product in the log domain, where
products become sums — numerically better behaved, and the reason the messages in
[section 1.2](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#12-messages) are additive. MASDA is the max-sum member of the
family; see [SPADA](#spada) for the sum-product one.

### Messages: responsibility and availability

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

### Belief

The score a variable ends up with once both message directions are combined:
$$b_{ij} = \beta_{ij} + \rho_{ij} - s_{ij}$$. It measures an edge's advantage over
its competitors, which is exactly why its *sign* carries no information about
whether to associate at all — the mistake [section 3.1](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#31-reading-out-the-answer)
is about. The belief is used as an ordering; $$\lambda$$ decides membership.

### Damping

Blending each new message with the previous one,
$$x^{(t+1)} \leftarrow (1-\eta) x_{\text{target}} + \eta x^{(t)}$$, to stop
oscillation. Standard practice in loopy BP and not optional here: undamped max-sum
on heavily tied problems plateaus instead of settling. Measured on this problem,
quality is flat for $$\eta$$ between 0.3 and 0.5, and everything in this series
runs at 0.4 (0.6 for the ordering experiment of [section 6.1](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#61-measured-dense-it-works-and-it-costs-more-than-it-returns),
which adds loops).

### MAP estimate

Maximum a posteriori: the single most probable joint configuration, as opposed to
the per-variable marginals. Stereo wants one disparity per pixel, so MAP is the
right target and max-sum is the right algorithm.

### Convergence and the anytime property

*Convergence* here means the messages stop changing. *Anytime* means the algorithm
can be stopped at any iteration and still return a usable answer, improving with
time. The most useful thing this project measured is that the two are far apart:
the messages are still moving at iteration thirty while the decision stopped moving
around iteration two, so the shipping configuration runs two iterations
([section 4.3](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#43-two-iterations-against-thirty)). Being anytime is also the
property an exact assignment solver lacks.

## The association problem

### Data association

The problem of deciding *which observation belongs to which thing*: measurements to
tracked objects in a radar, detections to identities in a tracker, left pixels to
right pixels in stereo. What makes it a problem rather than a lookup is that the
answer is constrained jointly — one measurement cannot belong to two objects — so
the decisions cannot be made independently per measurement.

### MASDA

Max-Sum Algorithm Data Association: max-sum loopy belief propagation on the
association factor graph, which is what this series is about. The derivation — the
factor graph, the exclusivity constraints, the closed-form messages and the
complexity argument — is in the [original MASDA
post](https://www.mariolueder.com/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA/).
This part reuses those equations unchanged and substitutes stereo's nouns
([section 1](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#1-dense-stereo-as-a-data-association-problem)).

### SPADA

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

### Clutter and misdetection

The two outside options that keep the association problem honest. *Clutter*
($$\lambda$$) is a measurement that belongs to nothing — in stereo, a left pixel
whose surface is not visible on the right. *Misdetection* ($$\gamma$$) is an object
that no measurement found — a right pixel hidden from the left camera. Both are
first-class variables in the factor graph rather than thresholds applied
afterwards, which matters because [occlusion](#occlusion) makes 10–20% of pixels
unmatchable on these scenes. On the score scale of [section 1.3](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#13-the-score)
$$\lambda = \gamma = -0.1$$ reads as "reject anything worse than a tenth of the way
from chance to perfect".

### One-to-one constraint

Also *uniqueness* or *mutual exclusivity*: at most one association per measurement
and at most one per object. It is physically true in stereo, because one surface
point projects once into each image. Enforcing it inside the inference is what this
whole series is testing, and [section 4.1](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#41-what-uniqueness-is-worth) prices it
at +10.7 points of precision over [winner-take-all](#winner-take-all) on identical
scores.

### Linear assignment problem

The combinatorial problem of choosing a maximum-weight one-to-one matching in a
bipartite graph — association with the uniqueness constraint and nothing else. It
is solvable exactly in polynomial time, which is why it is the right yardstick for
an approximate method. MASDA stops being a LAP the moment a factor is added that
couples two associations, such as the [ordering constraint](#ordering-constraint) —
that is the argument of [section 5](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#5-speed-the-representation-decides-it).

### Jonker-Volgenant and the Hungarian method

The exact LAP solvers. Kuhn's Hungarian method is the classical one; the
Jonker-Volgenant shortest-augmenting-path algorithm is the fast modern variant and
the one behind `scipy.optimize.linear_sum_assignment`. Used here as ground truth
for the *optimisation*, separately from ground truth for the *answer*:
[section 4.2](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#42-against-the-exact-optimum) runs per-row JV over every row of two
scenes and finds MASDA's precision indistinguishable from exact.

> H. W. Kuhn (1955). *The Hungarian method for the assignment problem.* Naval
> Research Logistics Quarterly 2(1-2), 83-97.
> [doi:10.1002/nav.3800020109](https://doi.org/10.1002/nav.3800020109)

> R. Jonker, A. Volgenant (1987). *A shortest augmenting path algorithm for dense
> and sparse linear assignment problems.* Computing 38(4), 325-340.
> [doi:10.1007/BF02278710](https://doi.org/10.1007/BF02278710)

### LP relaxation and the uniqueness condition

Relax the binary association variables to $$[0,1]$$ and the LAP becomes a linear
program. Bayati, Shah and Sharma proved max-product correct on bipartite matching
*provided that LP has a unique optimum* — and a tie between two candidate scores is
precisely how uniqueness fails. This is why the degenerate case in
[section 3.1](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#31-reading-out-the-answer) is not a bug to patch: it is where the
guarantee stops. On an aggregated Census volume, which quantises to few levels,
exact ties are routine, so the condition fails on roughly half the rows
([section 4.2](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#42-against-the-exact-optimum)) — measured here, not predicted by
the theory.

> M. Bayati, D. Shah, M. Sharma (2008). *Max-Product for Maximum Weight Matching:
> Convergence, Correctness, and LP Duality.* IEEE Transactions on Information
> Theory 54(3), 1241-1251.
> [doi:10.1109/TIT.2007.915695](https://doi.org/10.1109/TIT.2007.915695)

### Objective ratio

The total score of MASDA's assignment divided by the exact optimum's, per row —
1.0 meaning "MASDA found an optimal assignment". Reported alongside precision in
[section 4.2](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#42-against-the-exact-optimum) because the two disagree so
instructively: closing an 0.8% objective gap buys no precision at all, which makes
the objective a demonstrably loose proxy for the quantity a consumer experiences.

### Greedy decode

The read-out rule: order candidates by [belief](#belief), accept a pair when both
sides agree and the score beats $$\lambda$$, then fill in what remains greedily by
belief. Not cosmetic — under near-ties every row's best belief points at the same
column, so mutual agreement alone commits one pair per row, and every greedily
accepted edge still raises the objective ([section 3.1](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#31-reading-out-the-answer)).

### Margin and the margin gate

The margin is best-minus-second-best in the same message currency: a per-pixel
confidence that MASDA produces as a by-product rather than as an extra pass. The
*margin gate* refuses to answer where the margin is small, trading coverage for
precision. It is the mechanism behind the 76% coverage in Part 2's SGM comparison,
and the reason a MASDA answer can be *absent* rather than merely wrong.

### Segment reduction and max-excluding

Both message updates need "the maximum over this row except column $$j$$", which is
quadratic if computed per element and constant time if the largest and
second-largest are cached. On an edge list the same thing is a *segment reduction*
— a scatter-max over the edges grouped by endpoint (`np.maximum.at` in NumPy),
plus the count of elements attaining the maximum so ties are handled rather than
ignored. This is the whole trick that makes the sparse form $$O(E)$$;
[section 3.2](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#32-the-sparse-matrix-form--this-is-the-design-not-an-optimisation)
gives the ten lines.

### Ordering constraint

Along a rectified scanline, correspondences of an opaque surface preserve left-right
order: matches should not cross. Scanline dynamic programming gets this for free —
it is the classical argument for DP-based stereo, going back to Ohta and Kanade —
and a plain assignment formulation does not, which makes it the standing objection
to using MASDA here. [Section 6](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#6-can-masda-express-the-ordering-constraint)
derives it as a pairwise factor costing one clamped scalar per conflicting edge,
verifies the closed form against brute-force max-sum, and then measures it as a
real but poor trade on the dense problem. Note that thin foreground objects
genuinely violate ordering, which is why $$\kappa$$ stays finite and why DP methods
need forbidden-move exceptions.

> Y. Ohta, T. Kanade (1985). *Stereo by Intra- and Inter-Scanline Search Using
> Dynamic Programming.* IEEE TPAMI 7(2), 139-154.
> [doi:10.1109/TPAMI.1985.4767639](https://doi.org/10.1109/TPAMI.1985.4767639)

### Fenwick tree

A binary indexed tree: prefix sums and prefix counts in $$O(\log n)$$ with an array
and no pointers. It is the standard way to count crossing pairs without enumerating
them, and in [section 6.1](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#61-measured-dense-it-works-and-it-costs-more-than-it-returns)
it turns from an optimisation into a precondition — with ~2400 crossing pairs per
row, the quadratic enumeration is the cost.

> P. M. Fenwick (1994). *A new data structure for cumulative frequency tables.*
> Software: Practice and Experience 24(3), 327-336.
> [doi:10.1002/spe.4380240306](https://doi.org/10.1002/spe.4380240306)

### Sinkhorn, optimal transport and SuperGlue

The soft-assignment cousin of this formulation. Optimal transport asks for a
one-to-one-ish coupling between two distributions; adding an entropy term makes it
solvable by Sinkhorn's alternating row-and-column normalisation, and *dustbin* rows
and columns give unmatched items somewhere to go — which is $$\lambda$$ and
$$\gamma$$ under another name. Max-sum is the zero-temperature limit of that
relaxation. SuperGlue is the influential learned instance: same structure, but the
scores come from an attention network and the dustbin costs are learned rather than
hand-set, which is exactly the weakness [section 9](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#9-what-would-improve-it)
admits to.

> M. Cuturi (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal
> Transport.* NeurIPS. [arXiv:1306.0895](https://arxiv.org/abs/1306.0895)

> P.-E. Sarlin, D. DeTone, T. Malisiewicz, A. Rabinovich (2020). *SuperGlue:
> Learning Feature Matching with Graph Neural Networks.* CVPR.
> [arXiv:1911.11763](https://arxiv.org/abs/1911.11763)

## Stereo vision

### Rectification and epipolar geometry

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

### Disparity

The horizontal offset between a point's two projections, $$d = x_L - x_R$$, in
pixels. It is inverse depth: $$Z = fB/d$$ for focal length $$f$$ and baseline
$$B$$, so one disparity step is a large distance change far away and a small one up
close. $$D$$ throughout this series is the number of disparities searched (60 on
the camera, up to 80 on the benchmark scenes), and a *disparity plane* is the
whole image scored at one fixed $$d$$.

### Occlusion

A surface visible in one camera and hidden in the other, so a correct match simply
does not exist. It affects 10–20% of pixels on these scenes, concentrated at depth
discontinuities — which is where stereo is hardest anyway — and it is the reason
[clutter and misdetection](#clutter-and-misdetection) have to be part of the model
rather than a threshold bolted on afterwards.

### Cost volume

The $$W \times H \times D$$ array holding a matching score for every (pixel,
disparity) pair: 40 MB per frame at 450×375, 98 MB at 848×480. Building it,
aggregating it, then reading it back to pick winners is the textbook four-stage
pipeline codified in Scharstein and Szeliski's taxonomy. Part 2's second section is
about never materialising it — with two candidates per pixel, the running top-2
*is* the reduced volume — and Part 3 fuses away even the intermediate planes.

> D. Scharstein, R. Szeliski (2002). *A Taxonomy and Evaluation of Dense Two-Frame
> Stereo Correspondence Algorithms.* IJCV 47(1-3), 7-42.
> [doi:10.1023/A:1014573219977](https://doi.org/10.1023/A:1014573219977)

### Census transform

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

### Hamming distance and popcount

The distance between two Census descriptors is the number of differing bits: XOR
them and count the ones. `popcount` is that count in hardware —
`__builtin_popcountll` becomes one instruction on x86-64 and a short sequence
around NEON's `cnt` on ARMv8 — which is what makes a 48-bit descriptor comparison
essentially free. Two unrelated descriptors agree on half
their bits by chance, which is the zero point the score scale of
[section 1.3](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#13-the-score) is built around.

### Truncated absolute difference

The absolute intensity difference between two pixels, clipped at a maximum. On its
own it is a weak matching cost; added to Census it contributes a *graded* signal
where Census is nearly saturated, which Part 2 measured at 10.3% → 9.7% bad-1.0 by
itself. Combining a binary descriptor with an intensity term this way is
established practice — ADCensus is the well-known instance.

> X. Mei, X. Sun, M. Zhou, S. Jiao, H. Wang, X. Zhang (2011). *On building an
> accurate stereo matching system on graphics hardware.* ICCV Workshops.
> [doi:10.1109/ICCVW.2011.6130280](https://doi.org/10.1109/ICCVW.2011.6130280)

### Cost aggregation

Summing or filtering each candidate's score over a neighbourhood, so the decision
rests on a region's evidence rather than one pixel's. It is the third stage of the
classical taxonomy and, on this project, the single largest accuracy lever: one
7×7 Census comparison quantises to 49 levels, and Part 2 measured unaggregated
scores at 28.1% bad against 12.7% for SGM stripped of its smoothness term
entirely. In factor-graph language, better unaries beat a cheap pairwise term —
which is also why the cheap smoothness factor is a recorded negative result.

### Edge-aware recursive filter

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

### Smoothness prior

A pairwise term that penalises disparity differences between neighbouring pixels,
on the argument that surfaces are mostly continuous. It is the mechanism
[SGM](#semi-global-matching) is built on, and the one this factor graph does *not*
have. Two cheap versions were tried here and are recorded as negatives — worse at
every weight — with the mechanism: on a quantised Census volume the information a
neighbour smoothness term would add is better added by
[aggregating the unaries](#cost-aggregation). The interesting unbuilt object is a
graph carrying both uniqueness and path smoothness at once
([section 9](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#9-what-would-improve-it)).

### Keypoints and detector repeatability

A *keypoint* is a distinctive image location found by a detector (FAST, Harris,
ORB) and summarised by a descriptor, so matching considers only a few hundred
points per image instead of every pixel. *Repeatability* is the fraction of
keypoints found in one image that the detector also finds in the other — and it
bounds recall before any matcher runs, since a point detected in only one view
cannot be matched. This series began as a sparse-keypoint study and moved to the
dense problem when that bound was measured at under 51% on this camera: the dense
formulation deletes the detector, and with it the ceiling. Parts 1 and 2 refer back
to "the original keypoint study" in that sense.

### Winner-take-all

Take each pixel's best-scoring disparity and stop — no uniqueness, no interaction
between pixels. It is the baseline every stereo matcher has to beat, and the
comparison that isolates what the [one-to-one constraint](#one-to-one-constraint)
contributes, because it can be run on *identical* scores
([section 4.1](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#41-what-uniqueness-is-worth)): 0.776 precision against MASDA's
0.884.

### Semi-global matching

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

### Left-right consistency check

Match left-to-right, match right-to-left, and keep only the pixels where the two
agree. The standard way to detect occlusions and mismatches after the fact, and the
standard way to get uniqueness without modelling it — at the price of running the
matcher twice. MASDA gets the same property inside one inference pass.

> P. Fua (1993). *A parallel stereo algorithm that produces dense depth maps and
> preserves image features.* Machine Vision and Applications 6(1), 35-49.
> [doi:10.1007/BF01212430](https://doi.org/10.1007/BF01212430)

### Sub-pixel disparity

Fitting a curve to the scores around the winning integer disparity to recover a
fractional answer. It is the refinement stage of the classical taxonomy and it
matters more for metric depth accuracy than anything else downstream — and it is
not implemented here, which [section 9](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#9-what-would-improve-it) records as an
open item rather than a detail.

### Coarse-to-fine

Solve at half resolution, upsample the answer, and search only a narrow band around
it at full resolution — a pyramid strategy, and the family every fast published CPU
matcher belongs to in some form. Part 2 measured the ceiling honestly (81.5% of
known pixels keep the truth inside a ±2 band) and then measured the delivery: 5.2×
less arithmetic, 1.2–1.4× faster on the desktop, and *flat* on the Jetson, for
three specific reasons that are the most useful negative result in the series.

### Repetitive texture and ambiguity

A periodic pattern produces several near-equal scores one period apart, so the
evidence genuinely does not identify the answer. This is the failure mode that
bounds every method on these scenes including the exact solver — uniqueness can
rule out a *conflict*, but a patch matched one period off conflicts with nothing.
It is also why the gain from uniqueness is largest exactly where the scene is worst
([section 4.1](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#41-what-uniqueness-is-worth)).

### Disparity-major and disparity-minor layout

Two ways to index the cost volume: `[d][x]` keeps whole constant-disparity planes
contiguous, `[x][d]` keeps each pixel's run of disparities contiguous. The choice
is not a style question but a memory-system question, and Parts 2 and 3 answer it
in *opposite* directions — the CPU wants planes resident in L2, the GPU wants the
innermost index to be the one a warp spans. Part 3's section 4 is the resolution:
neither layout is wrong; trying to make one implementation serve both was.

## Datasets and metrics

### Middlebury stereo datasets

The standard rectified stereo benchmark with dense ground truth. This series uses
Teddy and Cones from the 2003 set and the six 2005 scenes (Art, Books, Dolls,
Laundry, Moebius, Reindeer) at third size, whose [dataset
page](https://vision.middlebury.edu/stereo/data/) grants permission to use and
publish the images and disparity maps. The images are not redistributed in this
repository; [section 2](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#2-ground-truth) also documents the disparity scale factor
of 3 being *established* by an identity check rather than assumed, because the 2005
archives do not document it.

> D. Scharstein, R. Szeliski (2003). *High-accuracy stereo depth maps using
> structured light.* CVPR, 195-202.
> [doi:10.1109/CVPR.2003.1211354](https://doi.org/10.1109/CVPR.2003.1211354)

> D. Scharstein, C. Pal (2007). *Learning conditional random fields for stereo.*
> CVPR. [doi:10.1109/CVPR.2007.383191](https://doi.org/10.1109/CVPR.2007.383191)

### Structured-light ground truth

How that ground truth was made: project coded light patterns onto the scene so
every surface point identifies itself, and the correspondence becomes
unambiguous — accurate to about a quarter pixel here, with a few percent of pixels
left unknown. Those unknowns are marked as disparity zero rather than by a separate
mask, which is why [section 2](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#2-ground-truth) excludes them from precision
instead of scoring them as wrong.

### Coverage, precision and the bad pixel rate

Three numbers that have to be read together. *Coverage* is the fraction of pixels
the matcher answers at all — a matcher with a [margin gate](#margin-and-the-margin-gate)
answers fewer. *Precision* is the fraction of answered pixels with known ground
truth that are correct within a tolerance (1 px throughout). *bad-1.0* is the
complement, the percentage wrong by more than 1 px, and it is the metric Part 2
reports because it is what the stereo literature reports. The product,
coverage × (1 − bad), is *correct-over-known*: the share of all knowable pixels
actually delivered correctly. Comparing two matchers on one of the three alone is
how a matcher that mostly abstains looks excellent.

### Interleaved best-of-N

The timing protocol on the Jetson: run the variants alternately within one session
and take each one's minimum, rather than running all of A then all of B. The board's
run-to-run variance is 37% at locked clocks and stable temperature, because six
heterogeneous cores get scheduled differently each run, so a single measurement is
worthless and consecutive blocks let drift masquerade as an effect.

## The matchers and methods this one is measured against

### ELAS

Efficient Large-Scale Stereo Matching: match a sparse set of robust support points
first, triangulate them into a prior, and search only near the resulting surface.
The canonical "don't sweep the whole disparity range" CPU matcher, and the direct
ancestor of the [coarse-to-fine](#coarse-to-fine) experiment Part 2 measures.

> A. Geiger, M. Roser, R. Urtasun (2011). *Efficient Large-Scale Stereo Matching.*
> ACCV 2010, LNCS 6492, 25-38.
> [doi:10.1007/978-3-642-19315-6_3](https://doi.org/10.1007/978-3-642-19315-6_3)

### PatchMatch stereo

Avoids the sweep from the other direction: instead of testing every disparity,
propagate good hypotheses — including slanted support planes — between neighbouring
pixels and refine them randomly.

> M. Bleyer, C. Rhemann, C. Rother (2011). *PatchMatch Stereo — Stereo Matching
> with Slanted Support Windows.* BMVC.
> [doi:10.5244/C.25.14](https://doi.org/10.5244/C.25.14)

### rSGM

Semi-global matching engineered for the CPU: SIMD throughout, fewer paths, and
subsampled far disparities, reaching VGA at 128 disparities above 16 Hz on a
desktop CPU. The reference point for "how fast should a careful CPU stereo matcher
be".

> M. Spangenberg, T. Langner, S. Adfeldt, R. Rojas (2014). *Large scale
> Semi-Global Matching on the CPU.* IEEE Intelligent Vehicles Symposium.
> [doi:10.1109/IVS.2014.6856419](https://doi.org/10.1109/IVS.2014.6856419)

### ReS2tAC

Real-time SGM on exactly the hardware class this project targets — embedded ARM
with NEON, and CUDA on Jetson-class GPUs. Two of its design decisions were
re-derived independently here before being recognised: putting *disparity* in the
SIMD lanes rather than pixels (Part 2's NEON kernel) and the same assignment for
CUDA warps (Part 3's layout). It is the closest published prior art to this
pipeline's engineering, on both processors.

> B. Ruf, J. Mohrs, M. Weinmann, S. Hinz, J. Beyerer (2021). *ReS2tAC — UAV-Borne
> Real-Time SGM Stereo Optimized for Embedded ARM and CUDA Devices.* Sensors
> 21(11), 3938. [doi:10.3390/s21113938](https://doi.org/10.3390/s21113938)

### ESPReSSo

Slanted PatchMatch made real-time for spacetime stereo, with edge-aware
aggregation under shared plane hypotheses — which is why Part 2 lists it among the
work behind this pipeline's aggregation stage.

> H. Nover, S. Achar, D. B. Goldman (2018). *ESPReSSo: Efficient Slanted
> PatchMatch for Real-Time Spacetime Stereo.* 3DV.
> [doi:10.1109/3DV.2018.00072](https://doi.org/10.1109/3DV.2018.00072)

### GPU-efficient recursive filtering

The standard way to parallelise a recurrence on a GPU: algebraically reassociate it
into blocks so the blocks can run concurrently. Part 3 could not use it — the
reassociation does not reproduce the CPU's integer truncation bit-for-bit, and
bit-identity was the referee — and the entry is here because that paper's existence
is also the explanation of why the warp-serial alternative loses.

> D. Nehab, A. Maximo, R. S. Lima, H. Hoppe (2011). *GPU-Efficient Recursive
> Filtering and Summed-Area Tables.* ACM TOG 30(6), SIGGRAPH Asia.
> [doi:10.1145/2024156.2024210](https://doi.org/10.1145/2024156.2024210)

## Hardware and implementation

### Jetson TX2

The target board: an NVIDIA Tegra X2 module with six ARM cores, a 256-core Pascal
GPU, and one LPDDR4 memory system shared between them. Everything in Parts 2 and 3
that is surprising traces back to that last clause — the CPU and the GPU compete
for the same bandwidth, and the board's memory streams are its weakest point
relative to a desktop. [NVIDIA's developer
page](https://developer.nvidia.com/embedded/jetson-tx2) is the specification.

### A57 and Denver cores

The six cores are not identical: four ARM Cortex-A57s and two of NVIDIA's own
Denver cores, a wider design with dynamic code optimisation. Heterogeneity is why
run-to-run variance is 37% ([interleaved best-of-N](#interleaved-best-of-n)) and
why thread pinning has *opposite* signs in the two tools — Part 3 measures pinning
the solve to the A57 cluster as essential for the GPU pipeline and 40% worse for the
CPU-only matcher, which needs the Denvers' throughput.

### RealSense D435 and the IR pair

The camera this project is built around: an Intel RealSense D435, used not for its
own depth output but for its two infrared cameras as a raw stereo pair at
848×480 — which is where that resolution, and the 33.3 ms frame budget, come from.
[Product page](https://www.intelrealsense.com/depth-camera-d435/).

### NEON and SIMD

Single Instruction Multiple Data: one instruction operating on several values at
once. NEON is ARM's 128-bit version — eight `int16` lanes, against AVX2's 256 bits
on the desktop, which is why halving the data type was worth 20% on the Jetson and
nothing on the desktop. `vqtbl4q_u8` is the NEON table-lookup instruction across
four registers that made the score loop's two lookup tables vector operations.
[ARM's intrinsics reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
is searchable per instruction.

### Q14 fixed point

Integer arithmetic standing in for fractions: an `int16` holding a value scaled by
$$2^{14}$$, so $$[-2, 2)$$ is representable to about $$6 \times 10^{-5}$$. The
score is naturally in $$[-1, 1]$$, so it fits with headroom to spare, and the whole
pipeline stays integer — which is what makes bit-identity between CPU and GPU
achievable at all, since integer arithmetic has no reassociation freedom.

### Warp, coalescing and shuffle

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

### Constant and shared memory

Two of the GPU's addressable memories. `__constant__` is optimised for *broadcast*
— all lanes reading one address — and serializes into replays when lanes read
different addresses, which is what 32 different Hamming distances per warp do by
construction. Shared memory is per-block scratch with no such penalty, and moving
the lookup tables there was worth ~2 ms in Part 3.

### Pinned memory and I/O coherency

*Pinned* (page-locked) host memory is what a GPU DMA engine can copy without the
CPU's involvement, and on discrete GPUs it is also fast for the CPU to read. On the
TX2 it is not: the SoC has no I/O coherency, so every `cudaHostAlloc` flavour —
including the one whose name suggests otherwise — is uncached from the CPU's side,
and a solver reading candidates out of it runs about seven times slower than from
ordinary pageable memory. The fix is a staged copy into a plain `std::vector`, and
NVIDIA's [CUDA for Tegra
notes](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html) document the
memory-coherency model this violates.

### Kernel fusion

Merging two passes so the intermediate never reaches memory. The principle Part 3
states — *a pass that stores exactly what the next pass reads is a fusion
candidate, and a store nothing reads afterwards is a bug you are paying for* —
found time three times on the GPU and *nothing* on the CPU, where the intermediate
plane was already L2-resident. Same transformation, opposite verdict — the series'
thesis in one experiment.

### Bandwidth bound

A kernel is bandwidth bound when its runtime is set by bytes moved rather than
arithmetic performed. Dividing each kernel's bytes by its measured time and
comparing against what the board actually achieves (~35–40 GB/s here) turns
optimisation from guesswork into arithmetic: a kernel at 19 GB/s is leaving half
the machine idle, and it also says when to *stop*. The roofline model is the formal
version of this reasoning.

> S. Williams, A. Waterman, D. Patterson (2009). *Roofline: an insightful visual
> performance model for multicore architectures.* Communications of the ACM 52(4),
> 65-76. [doi:10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785)

### Amdahl's law

The speedup available from parallelising a program is capped by the part that stays
serial. Quoted here because it was the answer after five wrong guesses: the workers
were 5.99 of 6 cores busy while alive, and the missing time was a serial prologue
and a serial-ish merge *outside* the pool. The instrument that showed it — per-thread
span printed next to per-thread busy — is the transferable part.

> G. M. Amdahl (1967). *Validity of the single processor approach to achieving
> large scale computing capabilities.* AFIPS Spring Joint Computer Conference.
> [doi:10.1145/1465482.1465560](https://doi.org/10.1145/1465482.1465560)

### Autovectorisation

The compiler turning a scalar loop into SIMD instructions by itself. It is worth
knowing about mainly because it is easy to destroy: putting a branchy insert inside
the recurrence loop cost 18% outright in Part 3's section 8. On a CPU, fuse loops
only if the hot loop stays branch-free.

### Bit-identity

Requiring two implementations to produce byte-identical output, checked with `cmp`.
Part 3 treats it as a design constraint rather than a test — every GPU intermediate
replicates the CPU's exact integer arithmetic — and it paid twice: a ten-pixel
tie-break error in 407,040 and a padding bug that silently broke six of eight
scenes are both invisible to an accuracy benchmark at the scale they occur. The
caveat is real too: `cmp` between two *multi-threaded* runs is not an identity
check, because dynamic work distribution changes which of two equal scores wins.
