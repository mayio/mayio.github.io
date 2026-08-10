---
layout: post
title: 'Dense Stereo Matching with Max-Sum Belief Propagation on a Jetson TX2 (MASDA, Part 2)'
subtitle: 'From the NumPy study to a shipping C++ matcher: what survives engineering, what it costs on embedded hardware, and why the cost volume is never built.'
thumbnail-img: /assets/img/2026-08-08-Dense-MASDA_files/maps_teddy.png
date: '2026-08-08 22:00:00 +0200'
categories: association
comments: false
mathjax: true
author: Mario Lüder
tags: [belief-propagation, data-association, computer-vision, embedded]
---

In [Part 1](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/) I
applied [MASDA][gl-masda] — [max-sum][gl-maxsum] [loopy belief propagation][gl-lbp] for
[data association][gl-assoc] — to sparse stereo
matching and measured what the [one-to-one constraint][gl-one2one] buys against ground
truth. It ended
with a list of things that would improve the matcher. This post is what happened when I
took Part 1's NumPy formulation — dense MASDA on sparse matrices — and made it a
shipping C++ matcher, on a desktop first and then on the [Jetson TX2][gl-tx2] it is
actually meant for.

*Every term this series uses is defined in the [series
glossary][gl-appendix] — factor graphs, Census, SGM, CUDA warps — with links to the
original papers. Terms link there on first use.*

The interesting part is not that it works. It is *which parts of MASDA survived contact
with 407,040 pixels per frame*, which parts had to change shape, and how often a
confident performance prediction died on contact with the hardware. I kept every
measurement, including the failures — especially the failures, because half of what
shipped came from an experiment that first said something I did not expect.

Results, briefly:

- Dense MASDA is **ahead of OpenCV's [SGM][gl-sgm] on accuracy** over eight
  [Middlebury][gl-middlebury] scenes with
  ground truth: **9.7% [bad-1.0][gl-metrics] against SGM's 10.9%**, at 76.0%
  [coverage][gl-metrics] against 78.0%.
- The same algorithm, the same [$$\lambda$$, $$\gamma$$][gl-clutter] and
  [message equations][gl-messages] as Part 1 —
  what changed is the *representation*, again. Part 1 measured 62× between dense
  matrices and the sparse edge list on identical math. The C++ matcher repeats the
  lesson at the next scale: the [cost volume][gl-costvolume] the textbook says to build
  is never materialised at all.
- Runtime went **246 → 29 ms** on the desktop over the course of this work (Teddy,
  450×375). At the camera's real 848×480 resolution it is **77 ms on the desktop and
  ~155 ms on the TX2** against a 33 ms real-time budget — not real-time on the target
  yet. The desktop-versus-TX2 comparison itself turned out to be one of the more
  instructive results (section 6).
- Removing the exhaustive [disparity][gl-disparity] sweep — the thing every fast
  published matcher does
  — was measured to be worth **5.2× of arithmetic** and delivered **1.0× of runtime** on
  the TX2. The reasons are specific and instructive, and they are the meat of section 6.
- Two failed sub-experiments reinforced Part 1's central finding from a new direction:
  **the candidate set decides the outcome**. Widening it helped even when the widening
  looked unprincipled; narrowing it helped even when the narrowing looked principled.

Everything here can be regenerated: the matcher is one C++ file, the benchmark is one
script against Middlebury ground truth, and every timing on the TX2 is an
[interleaved best-of-N][gl-bestofn] because the board's run-to-run variance is 37% and
single runs there mean nothing.

*[Part 3](https://www.mariolueder.com/2026-08-09-Realtime-Dense-MASDA-on-the-Jetson-GPU/) takes
the step this post ends on: the image plane moves to the TX2's GPU, the CPU keeps
the MASDA solve, and the matcher runs at **28.9 ms per frame at 848×480 — 34.6 Hz**,
bit-identical to this implementation.*

---

## 1. From the study to the product: what stays, what changes

Part 1 formulates the row problem: every left pixel $$(y, x)$$ is a measurement,
its candidate "objects" are the right pixels $$(y, x-d)$$ within the disparity
range, [clutter $$\lambda$$ and misdetection $$\gamma$$][gl-clutter] are the outside
options, and
the candidates live in a sparse matrix.

The message equations do not change. The max-sum [messages][gl-messages] $$\rho$$ (over a
left pixel's
alternatives) and $$\beta$$ (over a right pixel's claimants) keep the closed form from
Part 1, the [damping][gl-damping] stays at 0.4, and — this surprised me — **two
iterations are
enough** in the engineered version, against the study's 30. The graph is so
regular that information does not have far to travel.

What changes is everything around the equations:

**The unary score is [aggregated][gl-agg], not raw.** A single 7×7
[Census][gl-census] comparison gives 49
[Hamming][gl-hamming] levels, and measured against SGM that quantisation — not the
missing [smoothness prior][gl-smooth] — is where the accuracy goes: SGM stripped of its
smoothness term *and* its
post-filtering still reached 12.7% bad against my unaggregated 28.1%. So the score for
(pixel, disparity) is Census plus a [truncated absolute difference][gl-tad] (a graded
cost, 10.3%
→ 9.7% bad-1.0 by itself), aggregated over an edge-aware support region by a
[recursive filter][gl-rf]
that stops at intensity edges. MASDA then runs on the aggregated scores. In
[factor-graph][gl-factorgraph] terms: better unaries beat a cheap pairwise term. I tried
the cheap smoothness factor early on and it was worse at every weight; aggregation is
where that information actually enters.

**The structure is a regular grid, so the edge list disappears again.** On a grid,
["max over this left pixel's other candidates"][gl-segred] is a contiguous run of $$D$$
scores, and "max
over this right pixel's other claimants" is a strided walk — stride exactly $$D+1$$.
Part 1 measured the same messages at 62× between dense matrices and the edge list;
the C++ form replaces the edge list with pointer arithmetic on a regular grid. Same
math, third representation.

**Rows are independent**, because correspondences live on a [rectified][gl-rect] row.
Every row is
solved by an independent MASDA instance, which is what makes the whole thing
embarrassingly parallel — and later, a GPU kernel rather than a rewrite.

## 2. The cost volume is never built

The textbook pipeline materialises a $$W \times H \times D$$ [cost volume][gl-costvolume]
— 40 MB per
frame at 450×375, 98 MB at 848×480 — then aggregates it, then reads it back to pick
winners. I built that first. Then I measured what the solver actually consumes.

**How many candidates per pixel does MASDA need?** I swept $$k$$, the number of
top-scoring disparities kept per pixel, against the full volume:

| k | 1 | **2** | 3 | 4 | 8 | full D |
|---|---|---|---|---|---|---|
| bad-1.0 | worse | **best** | = | = | = | = |

Two. Not approximately two — *exactly* two, and $$k=2$$ was measurably better than
$$k=8$$, because the extra candidates are noise the solver has to argue with. This is
Part 1's precision-by-margin table speaking again: the second-best candidate carries
real information (it defines the margin), the eighth carries none.

With $$k=2$$, the running top-2 per pixel *is* the reduced volume. So the pipeline
computes one disparity plane at a time — score, filter — and inserts it into a per-pixel
top-2 list, and the 40 MB array never exists. The insert's common case is a rejection
that reads one cached plane, so the whole stage is one streaming pass over the scored
planes. Removing the volume was measured at **1.9× end-to-end**, and it is also what
freed the accuracy work: the aggregation radius, the graded cost and the margin gate
were all tuned after this, on a pipeline fast enough to iterate on.

The solver's outputs are unchanged from Part 1: a disparity per pixel where the
one-to-one assignment says so, and a **[margin][gl-margin]** — best minus second-best in
the same
message currency — which is the confidence the gate consumes. The margin gate trades
coverage for precision exactly as in the study.

## 3. Where it stands against SGM

Eight Middlebury scenes with ground truth (Teddy, Cones, Art, Books, Dolls, Laundry,
Moebius, Reindeer), pixel-pooled:

| | coverage | bad-1.0 | desktop 450×375 | TX2 450×375 | TX2 848×480 |
|---|---|---|---|---|---|
| OpenCV SGM | 78.0% | 10.9% | 16 ms | not measured | not measured |
| **dense MASDA** | 76.0% | **9.7%** | 39 ms | 70 ms | 152 ms |

(Runtimes are best-of-6 on Teddy and on a real [D435 IR pair][gl-d435]; the TX2 columns
are interleaved best-of-6 because that board's run-to-run variance is 37%. SGM's 16 ms is
OpenCV on the desktop; I have not built OpenCV on the TX2, so those cells are empty
rather than scaled — scaling desktop numbers to the Jetson is how this project
once got a figure wrong by 3×.)

Ahead on accuracy by 1.2 points, behind on coverage by 2.0, behind on runtime. The
accuracy is the part I care about here, because SGM is a strong,
heavily-engineered baseline and MASDA reaches past it with a *different mechanism*:
uniqueness plus confidence instead of path-wise smoothness. SGM has no uniqueness
constraint at all — it needs a [left-right consistency check][gl-lrc] bolted on
afterwards to get
to 78% coverage, which is two matcher runs. MASDA gets mutual exclusivity in the
inference itself, in one run, and produces a calibrated margin as a by-product.

The maps, all eight scenes — left image, ground truth, and the two variants this post
compares (section 6). Black is "no answer": either no ground truth, or the margin gate
declined to commit. Note how closely the two right columns agree; that agreement is
measured at parity in the tables below.

![teddy](/assets/img/2026-08-08-Dense-MASDA_files/maps_teddy.png)
![cones](/assets/img/2026-08-08-Dense-MASDA_files/maps_cones.png)
![Art](/assets/img/2026-08-08-Dense-MASDA_files/maps_Art.png)
![Books](/assets/img/2026-08-08-Dense-MASDA_files/maps_Books.png)
![Dolls](/assets/img/2026-08-08-Dense-MASDA_files/maps_Dolls.png)
![Laundry](/assets/img/2026-08-08-Dense-MASDA_files/maps_Laundry.png)
![Moebius](/assets/img/2026-08-08-Dense-MASDA_files/maps_Moebius.png)
![Reindeer](/assets/img/2026-08-08-Dense-MASDA_files/maps_Reindeer.png)

Where the residual error lives — absolute error against ground truth on Teddy, for both
variants. The error is at depth discontinuities and in the
[repetitive-texture][gl-ambiguity] regions,
which is exactly where Part 1's ambiguity analysis predicted every matcher, including
the exact solver, loses precision:

![teddy error](/assets/img/2026-08-08-Dense-MASDA_files/err_teddy.png)

## 4. The speed work, and the discipline it forced

The desktop journey was 246 → 39 ms across roughly twenty measured changes (29 with the coarse-to-fine mask of section 6). I will not
walk through all of them; the ones worth a paragraph each are the ones that generalise.

**The volume removal** (section 2): 1.9×, and the largest single step.

**int16 through the score, filter and top-2.** The score is $$(24 - \text{hamming})/24$$
in $$[-1, 1]$$, which maps exactly onto [Q14 fixed point][gl-q14] with headroom. Measured:
**neutral on the desktop, 20% on the Jetson** — half the [lanes][gl-lanes], half the
bandwidth, and
[NEON][gl-neon] is 128-bit against AVX2's 256. This number pair is the whole argument for
measuring
on the target: the desktop said "don't bother", the target said "20%".

**[Centre-symmetric census][gl-census]** (24-bit descriptors): worthless on the desktop,
10% on the
TX2, and it costs 1.2 points of bad-1.0 — a knob with a price tag on each side, recorded
and left off.

**A [NEON][gl-neon] kernel with disparity in the [vector lanes][gl-lanes].** The score
loop was the largest
single item and its scalar [popcount][gl-hamming] looked like the reason. The first
attempt vectorised
eight *pixels'* popcounts and bought nothing: the table lookups next to the popcount
stayed per-pixel gathers and took over the loop. The literature
([ReS2tAC, Ruf et al.][gl-res2tac])
puts *disparity* in the lanes instead — then the eight Hamming distances land in one
register and NEON's `vqtbl4q_u8` turns both lookup tables into vector operations. Built,
[bit-identical][gl-bitid] to the scalar loop, **1.59× on the score loop — and 0.92× on
the stage**,
because the eight-disparity work quantum costs more scheduler occupancy than the kernel
saves. It is in the tree behind a flag, off by default. A 1.6× on 30% of a stage is an
11% ceiling, and I had ranked it first by item size. Ranking by size × achievable ÷
collateral is the correction.

**The measurement discipline itself earned its keep.** Three rules, each paid for:

1. *The desktop is not a proxy for the TX2*, in either direction (int16, csct above).
2. *TX2 variance is 37%* at locked clocks and stable temperature —
   [four A57s and two Denvers][gl-cores] being scheduled differently run to run.
   Everything is [interleaved best-of-N][gl-bestofn].
3. *`cmp` between two multi-threaded runs is not an identity check.* Two identical runs
   differ in a handful of pixels because the top-2 insert keeps the first of two equal
   scores and work is handed out dynamically. Bit-identity is checked single-threaded;
   accuracy is checked with the benchmark. An earlier experiment was part-reverted on
   identity evidence that this rule retroactively voids.

## 5. Five wrong guesses about occupancy, then the instrument

At 848×480 — the camera's real resolution, 2.4× the pixels of the Middlebury scenes —
the TX2 measured **200 ms**, and the cost stage's CPU-to-wall ratio said "3.85 of 6
cores busy". I now know that number was misread, and the misreading cost three built-
and-reverted "fixes" earlier in this project and two more mechanism guesses this week
(an L2 working-set story and a load-imbalance story, both measured false).

The instrument that settled it: per-thread *span* (time the thread existed) printed next
to per-thread *busy* (sum of its in-loop timers). Result: **5.99 of 6 busy while
alive**. The workers were saturated; the missing time was *outside* the pool — a serial
prologue (allocation, coefficient computation) and a serial-ish merge pass between the
cost pool and the solve pool. [Amdahl][gl-amdahl], not scheduling.

Three fixes, measured together at **1.22×** (193 → 158 ms end-to-end, interleaved
best-of-6):

- **The merge fused into the solve.** The per-pixel 2-of-2n candidate selection was a
  separate threaded pass writing 8 MB of merged arrays that only the solver read. Done
  row-by-row inside the solver instead, the candidates land hot in the 28 KB the row
  solve reads anyway, and the pass, the arrays and their serial zero-fill all vanish.
- **The filter coefficients become a 512-byte table.** Two `exp()` per pixel, ~700k
  transcendentals — but the input is the difference of two bytes, which has 256 values.
- **A dead normalisation removed.** 407k float divisions for a buffer the shipping
  path never reads.

The general lesson I keep relearning on this project: **when two mechanism guesses fail
on the same code, stop guessing and add the instrument.** Every timer added this week
found money somewhere no story had pointed.

## 6. Removing the D-plane sweep: 5.2× of arithmetic, 1.0× of runtime

This is the comparison this post exists for: the full sweep (score all $$D$$ disparity
planes) against a [coarse-to-fine][gl-c2f] variant that only scores planes where a
half-resolution prior says the answer might be.

**The idea has a measured ceiling.** Run the matcher at half resolution (a quarter of
the pixels, half the disparities — an eighth of the work), upsample, and search only
$$\pm 2$$ around the coarse answer: if the truth is outside that band, no refinement can
recover it. Measured on ground truth, **81.5% of known pixels keep the truth in-band**,
against 67.9% [correct-over-known][gl-metrics] delivered by the full sweep — headroom, at
5.2× less
arithmetic. Every fast published CPU matcher lives on some version of this:
[ELAS][gl-elas]
narrows the search around triangulated support points,
[PatchMatch][gl-patchmatch] propagates hypotheses
instead of sweeping, [rSGM][gl-rsgm] subsamples the far disparities.

**The first construction failed, and the mechanism matters.** I indexed the fine-level
planes by *offset from the prior*, so the filter aggregated pixels whose absolute
disparities differ wherever the prior slopes — which is everywhere, and wildly at every
depth discontinuity. [Aggregation][gl-agg] is worth 16 points of bad-1.0 here, and this
broke it:
4 points worse *at every band width up to ±30*, where the band covers the whole range
and the only remaining difference is the indexing. A quantity that does not move when
its supposed cause is varied sevenfold is not a tuning problem.

**The second construction keeps the planes absolute** and uses the prior only as a
*mask*: plane $$d$$ is scored on the rows whose per-row interval wants it, filtered
over the plane's bounding rectangle (a rectangular variant of the recursive filter),
and inserted strictly inside the interval — so a wrong prior costs candidates, never
wrong values.

Accuracy, eight scenes, pixel-pooled:

| | coverage | bad-1.0 | correct-over-known |
|---|---|---|---|
| full sweep | 76.0% | 9.5% | 68.8% |
| **coarse-to-fine mask** | 76.7% | 10.5% | 68.6% |

Parity, and the maps above show the agreement. The loss concentrates where you would
predict: thin structures (Art: 12.7 → 14.9) that vanish at half resolution, so the
prior never proposes them.

Runtime, both machines, both resolutions, same binary and flags. The desktop is a
4-core x86 at best-of-6; the TX2 is six cores at 2.03 GHz, interleaved best-of-6:

| | desktop | TX2 | TX2 / desktop |
|---|---|---|---|
| 450×375, full sweep | 39 ms | 70 ms | 1.8× |
| 450×375, coarse-to-fine | **29 ms** | 71 ms | 2.5× |
| 848×480, full sweep | 90 ms | 152 ms | 1.7× |
| 848×480, coarse-to-fine | **77 ms** | 155 ms | 2.0× |

**A 1.2–1.4× win on the desktop and *flat* on the TX2 — at both resolutions.** The same
change, measured on two machines, has a different sign. This is the third time this
project's desktop has predicted the wrong outcome for the target (int16 was neutral
there and worth 20% on the TX2; centre-symmetric census was worthless there and worth
10% on the TX2; now the mask, in the opposite direction), and it is why every number in
this post says which machine it came from.

Where the platforms actually disagree — per stage, at 848×480, representative runs:

| stage | desktop, full | TX2, full | desktop, c2f | TX2, c2f |
|---|---|---|---|---|
| census | 5 ms | 10–14 | 5 | 8–12 |
| cost (score + filter + insert) | 72–90 | 104–110 | 54 | 64 |
| **solve (MASDA + merge)** | **12** | **33–48** | **11** | **46–50** |
| coarse level, whole | — | — | 17 | 43 |

The cost stage — the arithmetic — is only ~1.4× apart between the machines. **The solve
is ~4× apart**, because it is not arithmetic: it is the fused candidate merge streaming
the per-thread top-2 planes, 19.5 MB per pass at this resolution, and memory streams are
where the TX2 is weakest relative to a desktop. So the two machines disagree about which
stage is expensive. The desktop's frame is dominated by the cost stage, which the mask
shrinks; the TX2's is increasingly dominated by the solve and the fixed coarse level,
which the mask cannot touch. Same code, different bottleneck, opposite verdict. The 5.2× of arithmetic is real —
the fine score loop genuinely shrinks by the band fraction — and three costs ate it:

1. **Background planes have full-width rectangles.** A disparity visible both left and
   right of a foreground object gets a bounding box spanning the row, so the fill and
   the filter still touch nearly the whole image for most planes. Arithmetic scales
   with the band; *rectangles* do not.
2. **The solver does not scale with $$D$$ at all.** Its cost is per-pixel, it is ~4×
   more expensive on the TX2 than on the desktop (the stage table above), and at ~47 ms
   it is now the largest single item on the board — [bandwidth, not
arithmetic][gl-bandwidth]. The mask cannot touch it.
3. **The coarse level is a fixed 43 ms on the TX2** (17 on the desktop), almost exactly
   what the mask saves from the fine cost stage there.

Two sub-experiments inside this construction are worth recording because they are
*about MASDA*, not about stereo:

- **A second search band around the coarse level's runner-up candidate** — MASDA
  exports its second-best per pixel, and at thin structures the runner-up is often the
  missing surface. Measured: *worse everywhere* (pooled 10.6 → 11.2). Where the coarse
  level is ambiguous, its runner-up is the wrong period of a
  [repetitive texture][gl-ambiguity], and a
  band around it hands the fine solver a wrong surface it can aggregate into a
  confident answer.
- **A strict per-pixel band-membership test at insert time** — the "principled" version
  of the mask. Measured: *a full point worse* (10.6 → 11.6). The row-interval slack
  admits candidates a pixel's own band would exclude, and they help.

Part 1 measured the same thing from the other side: exact per-row assignment is no
more precise than MASDA on identical candidates (0.914 against 0.915 on Teddy), so
the inference is not what bounds the result — the candidate set is. Both arrows here point the same way: what you
offer the solver matters more than how cleverly you restrict it.

## 7. Where this leaves it

Where real-time on the TX2 stands, at the camera's 848×480:

| milestone | desktop 848×480 | TX2 848×480 | TX2 vs 30 Hz budget |
|---|---|---|---|
| measured at target resolution | — | 200 ms | 6.0× over |
| Amdahl fixes | 90 ms | **152 ms** | 4.6× over |
| coarse-to-fine mask | **77 ms** | 155 ms | flat |

(The Amdahl fixes also carried down to the smaller resolution: Teddy on the TX2 went
from ~89 ms to ~70 over the same changes, none of which were aimed at it.)

Dense MASDA on this CPU is a 6 Hz matcher today, or ~15 Hz at half resolution. The
solver itself — the part that is MASDA — is 47 of those milliseconds and is bound by
memory streaming, not by message-passing arithmetic; the identified next step on the
CPU is a row-stripe parallelisation that deletes the merge outright, at the price of
truncating the vertical filter at stripe boundaries. That is a measurable quality
question, and it is queued behind a bigger one.

**The bigger one is the GPU.** The TX2's GPU sits at load zero through all of this, and
every structural property that made the CPU implementation fast — independent disparity
planes, independent rows, a solver that is two [strided reductions][gl-segred] — is a
description of
a CUDA kernel. ReS2tAC runs SGM in real time on exactly this class of hardware; dense
MASDA has strictly more parallel structure than SGM, plus a property SGM lacks: it
already knows how confident it is. That is the next post.

The measurement discipline is the part I would keep even if the matcher were thrown
away. Five occupancy hypotheses were wrong before an instrument was right; the desktop
predicted the wrong sign for the target twice; the "obviously correct" 5.2× became
1.0× for reasons no amount of reading the code would have produced. Roughly half of
what shipped came from an experiment that first said something I did not expect —
which is, I think, the definition of an experiment worth running.

---

*The matcher is `de_dense.cpp` in the project's `core/`, plain C++14 with no
dependencies; the benchmark regenerates every number in this post from the Middlebury
scenes with one command. Part 1, with the derivation of the message equations, the
$$\lambda$$/$$\gamma$$ semantics and the sparse results, is
[here](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/).*

**References**

Full citations with DOIs, along with every term this post uses, are in the
[series glossary][gl-appendix].

- [Geiger, Roser, Urtasun, *Efficient Large-Scale Stereo Matching*][gl-elas] (ELAS),
  ACCV 2010 — support points + triangulated prior, the canonical "don't sweep" CPU
  matcher.
- [Bleyer, Rhemann, Rother, *PatchMatch Stereo*][gl-patchmatch], BMVC 2011 — hypothesis
  propagation instead of a sweep; slanted support windows.
- [Spangenberg et al., *Large Scale Semi-Global Matching on the CPU*][gl-rsgm], IV 2014 —
  VGA×128 disparities above 16 Hz on a CPU; disparity subsampling.
- [Ruf et al., *ReS2tAC — UAV-Borne Real-Time SGM Stereo Optimized for Embedded ARM and
  CUDA Devices*][gl-res2tac], Sensors 21(11), 2021 — the disparity-in-the-lanes NEON
  formulation and the embedded CUDA baseline.
- [Nover, Achar, Goldman, *ESPReSSo: Efficient Slanted PatchMatch for Real-Time Spacetime
  Stereo*][gl-espresso], 3DV 2018 — edge-aware aggregation under shared plane hypotheses.

[gl-appendix]: https://www.mariolueder.com/masda-glossary/
[gl-masda]: https://www.mariolueder.com/masda-glossary/#masda
[gl-maxsum]: https://www.mariolueder.com/masda-glossary/#max-sum-max-product-and-sum-product
[gl-lbp]: https://www.mariolueder.com/masda-glossary/#loopy-belief-propagation
[gl-assoc]: https://www.mariolueder.com/masda-glossary/#data-association
[gl-factorgraph]: https://www.mariolueder.com/masda-glossary/#factor-graph
[gl-messages]: https://www.mariolueder.com/masda-glossary/#messages-responsibility-and-availability
[gl-damping]: https://www.mariolueder.com/masda-glossary/#damping
[gl-clutter]: https://www.mariolueder.com/masda-glossary/#clutter-and-misdetection
[gl-one2one]: https://www.mariolueder.com/masda-glossary/#one-to-one-constraint
[gl-margin]: https://www.mariolueder.com/masda-glossary/#margin-and-the-margin-gate
[gl-segred]: https://www.mariolueder.com/masda-glossary/#segment-reduction-and-max-excluding
[gl-rect]: https://www.mariolueder.com/masda-glossary/#rectification-and-epipolar-geometry
[gl-disparity]: https://www.mariolueder.com/masda-glossary/#disparity
[gl-costvolume]: https://www.mariolueder.com/masda-glossary/#cost-volume
[gl-census]: https://www.mariolueder.com/masda-glossary/#census-transform
[gl-hamming]: https://www.mariolueder.com/masda-glossary/#hamming-distance-and-popcount
[gl-tad]: https://www.mariolueder.com/masda-glossary/#truncated-absolute-difference
[gl-agg]: https://www.mariolueder.com/masda-glossary/#cost-aggregation
[gl-rf]: https://www.mariolueder.com/masda-glossary/#edge-aware-recursive-filter
[gl-smooth]: https://www.mariolueder.com/masda-glossary/#smoothness-prior
[gl-sgm]: https://www.mariolueder.com/masda-glossary/#semi-global-matching
[gl-lrc]: https://www.mariolueder.com/masda-glossary/#left-right-consistency-check
[gl-c2f]: https://www.mariolueder.com/masda-glossary/#coarse-to-fine
[gl-ambiguity]: https://www.mariolueder.com/masda-glossary/#repetitive-texture-and-ambiguity
[gl-middlebury]: https://www.mariolueder.com/masda-glossary/#middlebury-stereo-datasets
[gl-metrics]: https://www.mariolueder.com/masda-glossary/#coverage-precision-and-the-bad-pixel-rate
[gl-bestofn]: https://www.mariolueder.com/masda-glossary/#interleaved-best-of-n
[gl-elas]: https://www.mariolueder.com/masda-glossary/#elas
[gl-patchmatch]: https://www.mariolueder.com/masda-glossary/#patchmatch-stereo
[gl-rsgm]: https://www.mariolueder.com/masda-glossary/#rsgm
[gl-res2tac]: https://www.mariolueder.com/masda-glossary/#res2tac
[gl-espresso]: https://www.mariolueder.com/masda-glossary/#espresso
[gl-tx2]: https://www.mariolueder.com/masda-glossary/#jetson-tx2
[gl-cores]: https://www.mariolueder.com/masda-glossary/#a57-and-denver-cores
[gl-d435]: https://www.mariolueder.com/masda-glossary/#realsense-d435-and-the-ir-pair
[gl-neon]: https://www.mariolueder.com/masda-glossary/#neon-and-simd
[gl-lanes]: https://www.mariolueder.com/masda-glossary/#vector-lanes
[gl-q14]: https://www.mariolueder.com/masda-glossary/#q14-fixed-point
[gl-bandwidth]: https://www.mariolueder.com/masda-glossary/#bandwidth-bound
[gl-amdahl]: https://www.mariolueder.com/masda-glossary/#amdahls-law
[gl-bitid]: https://www.mariolueder.com/masda-glossary/#bit-identity
