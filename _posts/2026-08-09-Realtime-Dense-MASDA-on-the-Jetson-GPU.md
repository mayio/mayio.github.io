---
layout: post
title: 'Real-Time Dense Stereo at 34 Hz on a Jetson TX2 (MASDA, Part 3)'
subtitle: 'The GPU takes the image plane, the CPU keeps the graph: architecture, dataflow, and why the memory layout the CPU rejected three times is exactly what the GPU wants.'
thumbnail-img: /assets/img/2026-08-09-Realtime-Dense-MASDA_files/progression.png
date: '2026-08-09 01:00:00 +0200'
categories: association
comments: false
mathjax: true
author: Mario Lüder
tags: [belief-propagation, data-association, computer-vision, embedded, cuda]
---

[Part 2](https://www.mariolueder.com/2026-08-08-Dense-MASDA-Belief-Propagation-Stereo-on-a-Jetson/)
ended on a diagnosis: the desktop and the [Jetson TX2][gl-tx2] disagree about which stage
of
the dense matcher is expensive, because they have different memory systems, and no
single-machine implementation can be right for both. This post draws the
conclusion: stop making one machine do both. The regular per-pixel work moves to
the TX2's GPU — which had sat at load zero through two days of CPU optimisation —
and the CPU keeps the part that is actually [MASDA][gl-masda]: the
[belief-propagation][gl-lbp] solve.

*Every term this series uses is defined in the [series glossary][gl-appendix], with
links to the original papers. Terms link there on first use.*

The result, measured as always as [interleaved best-of-N][gl-bestofn] on the target:

| | 848×480, D=60 | 450×375, D=60 |
|---|---|---|
| CPU only, six cores (Part 2) | 152 ms | 70 ms |
| GPU + CPU, pipelined | **28.9 ms — 34.6 Hz** | **12.6 ms — 79 Hz** |

Full quality — the 48-bit [Census][gl-census], the [graded cost][gl-tad], the
[edge-aware recursive filter][gl-rf], all 60 [disparities][gl-disparity], the
[one-to-one][gl-one2one] solve with its [margin gate][gl-margin] — and
**[bit-identical][gl-bitid] to the CPU implementation**: `cmp` on the output files finds
zero
differing bytes, on all eight ground-truth scenes and on a real camera pair,
including through thirty pipelined frames. Every accuracy number from Part 2
therefore carries over *by construction*: 9.7% [bad-1.0][gl-metrics] against
[SGM's][gl-sgm] 10.9%, without re-running a single benchmark.

![progression](/assets/img/2026-08-09-Realtime-Dense-MASDA_files/progression.png)

The chart is the honest shape of the work: the port itself bought 2.7×, and the
remaining 2× came from five specific findings about where the first port wasted
memory traffic, wasted issue slots, and waited. Sections 5–7 give the accounting.
Three attempts failed and are documented with mechanisms, because two of them are
the kind of thing I would otherwise try again next year.

---

## 1. The architecture: the GPU owns the image plane, the CPU owns the graph

The split is not "GPU does the slow parts." It follows the *structure* of the two
workloads:

- **The cost side is regular.** Census, the graded cost, the recursive filter and
  the top-2 selection touch every (pixel, disparity) pair in a pattern known at
  compile time. No data-dependent branching, no irregular memory. This is the
  textbook definition of GPU work.
- **The solve side is irregular.** MASDA's [messages][gl-messages] flow along a
  [factor graph][gl-factorgraph] whose
  structure depends on the candidates — claimant lists per right pixel, a
  [greedy decode][gl-greedy] over a sorted order. It is also *small*: with two
  candidates per pixel
  the whole row problem fits in 28 KB of cache, and six ARM cores handle the full
  frame in ~23 ms. Making this a GPU kernel would be work spent making the
  architecture uniform, not making the system faster.

The interface between the two is deliberately narrow: **two scored disparity
candidates per pixel** — the same $$k=2$$ result from Part 2, where keeping more
candidates measurably *hurt* accuracy. Eight megabytes per frame instead of a
forty-nine megabyte [cost volume][gl-costvolume]. The GPU reduces; the CPU decides.

```
   GPU (Pascal, 256 cores)                   CPU (4× A57, pinned)
  ┌────────────────────────────────┐        ┌──────────────────────────┐
  │ census L,R    (uint64/px)      │        │                          │
  │ filter coeffs (LUT per px)     │        │  MASDA row solver        │
  │ graded cost   (Census + AD)    │        │  ρ/β messages, 2 iters   │
  │ edge-aware recursive filter    │        │  one-to-one greedy decode│
  │ running top-2 per pixel        │        │  margin gate             │
  └───────────────┬────────────────┘        └────────────▲─────────────┘
                  │   candidates: 2 × (score, d) + count per pixel      
                  └──────────────  8 MB / frame  ────────┘
```

One Tegra-specific trap lives on that arrow, and it cost 300 ms twice before I
believed it: **the TX2 has [no I/O coherency, so every flavour of
`cudaHostAlloc` memory][gl-pinned] — including the one the documentation's mental model
calls
"cached pinned" — is uncached on the CPU side.** A solver reading candidates from
such memory runs seven times slower than from ordinary pageable memory. The
candidates travel through a staged `cudaMemcpy` into a plain `std::vector`, and
section 3 shows where that copy hides.

## 2. Dataflow: five kernels, two of which are fusions

Per frame, in stream order:

```
 frame ──► upload L,R (pinned staging, ~1 ms)
        ──► k_census        L,R → 48-bit descriptors        (1.5 ms, with coeffs)
        ──► k_rf_coeffs     |∇I| → filter coefficients a(x)
        ──► k_score_hfwd    census ──► score ──► horizontal  (8.8 ms)
        │                   FORWARD filter pass, fused:
        │                   the raw score never touches DRAM
        ──► k_hbwd          horizontal backward pass         (5.0 ms)
        ──► k_vert_fwd      vertical forward pass            (3.3 ms)
        ──► k_vert_bwd_top2 vertical backward pass with the  (7.4 ms)
        │                   TOP-2 fused in as a warp
        │                   reduction; its own stores are
        │                   DELETED — nothing reads them
        ──► fetch           candidates → cached host memory  (hidden, §3)
        ──► CPU solve       480 independent rows on 4 A57s   (23 ms, parallel)
```

The two [fusions][gl-fusion] are where most of the speed lives, and they follow one
principle worth stating because it found money three times:

> **A pass that stores exactly what the next pass reads is a fusion candidate. A
> store that nothing reads afterwards is a bug you are paying for.**

The first port scored the volume (write 47 MB), read it back to filter (94 MB),
wrote it filtered, read it again for top-2. The final version computes the score
*inside* the first filter pass and consumes the volume *inside* the last one — the
scored-but-unfiltered volume never makes a round trip, and the fully-filtered
volume is never materialised at all. What remains in DRAM is the minimum the
data dependencies allow: the recurrences genuinely need the intermediate planes.

## 3. The pipeline: where the CPU and GPU overlap

Single-frame latency is ~52 ms; the *throughput* is 28.9 ms/frame because the GPU
computes frame $$t{+}1$$ while the CPU solves frame $$t$$:

```
          ├────────── 28.9 ms steady state ──────────┤
 GPU      │ kernels, frame t+1        (~26 ms)  ▓▓▓▓▓│ kernels, frame t+2
 copy eng │        fetch t+1 ▒▒ (4 ms, hidden)       │
 CPU A57  │ solve frame t  ██████████ (23 ms)        │ solve frame t+1
 CPU main │ launch t+1 ▏               join, loop ▏  │
```

Two details make the overlap honest rather than hopeful:

- **The fetch runs on its own thread.** `cudaMemcpy` serializes with the stream,
  so the fetcher thread simply blocks until frame $$t{+}1$$'s kernels finish and
  then copies — *while the solve of frame $$t$$ is still running on the A57s*.
  Before this, the 4 ms copy sat serially in the loop.
- **The solve threads pin themselves to the A57 cluster.** The TX2 has
  [four A57s and two Denver cores][gl-cores]; unpinned, the scheduler wanders across both
  and the solve
  varied 30–45 ms run to run — the same 37% variance that forced interleaved
  best-of-N timing in Part 2. Pinned to the A57s it sits at 22.6–23.1 ms. The
  Denvers are left to the CUDA driver and the fetcher.

The measurement itself is part of the design: the pipelined mode reuses the same
pair for N frames, and the *last* frame's output is what gets written and
compared — so a broken overlap (a buffer raced, a fence missing) breaks the
bit-identity check instead of passing quietly.

## 4. The layout that ends the [d][x]-versus-[x][d] saga

Parts 1 and 2 kept running into the same question: is the cost volume indexed
[disparity-major (`[d][x]`, whole planes per disparity) or disparity-minor
(`[x][d]`, a run of disparities per pixel)][gl-layout]? The CPU answered three times:
disparity-major, because the aggregation filter wants whole constant-disparity
planes, and the `[x][d]` transpose was measured as the dominant memory cost of
the early implementation.

**The GPU wants the opposite, and now I can say precisely why.** The volume is
stored `vol[y][x][k]` — k innermost, padded to 64-aligned runs — and a
[warp][gl-warp] is
**32 consecutive disparities of one image row**:

- The right-census reads for 32 consecutive $$d$$ at one $$x$$ are 32
  *consecutive* addresses — one [coalesced][gl-warp] 256-byte window that slides one
  element
  per step and lives in L1. The left descriptor and the filter coefficient are
  the same address for all 32 lanes — a hardware broadcast.
- Every volume access anywhere in the pipeline is a k-run: an aligned 64- or
  128-byte transaction. Nothing strides.
- Each lane carries its own independent filter recurrence in registers. No lane
  waits on another; no shuffle is needed for the *filter* at all.

The same assignment appears in [ReS2tAC (Ruf et al. 2021)][gl-res2tac] for SGM —
disparity in
the lanes — and this is the third time this project has re-derived their design
point from a different direction. The resolution of the saga is that **neither
layout was wrong; each machine's memory system picks its own.** A cache hierarchy
with 512 KB of shared L2 wants one plane at a time, resident. A latency-hiding
machine with 32-wide transactions wants the innermost index to be the one the
warp spans. The two-day detour of trying to make one implementation serve both
was the actual mistake, and it is the most transferable lesson in the series.

## 5. Why it is fast, part I: work that no longer exists

The honest accounting of the 56.6 → 28.9 ms factor is mostly *deletions*:

| removed | was costing | mechanism |
|---|---|---|
| two 93 MB volume transposes | ~12 ms | k-minor layout: nobody needs the other orientation |
| raw-score round trip through DRAM | ~8 ms | score fused into the horizontal forward pass |
| separate top-2 kernel | 10.9 ms | fused into the vertical backward pass as a warp reduction |
| backward-pass stores | ~3 ms | with top-2 fused, the filtered volume has no reader — the stores are deleted outright |
| constant-memory replays in the score | ~2 ms | [`__constant__`][gl-cmem] serializes on divergent indexing, and 32 different Hamming distances per warp is divergent *by construction*; the tables moved to shared memory |
| half-width memory transactions | ~7 ms | every remaining pass works on k-pair `int32`s: full 128-byte lines |
| the fetch, serialized | ~4 ms | moved to a thread; hides inside the solve |
| solve variance | 10–20 ms of wander | Denver cores excluded by pinning |

[Bandwidth arithmetic][gl-bandwidth] kept the process honest: each kernel's bytes-moved
divided
by its measured time, compared against the ~35–40 GB/s this board actually
achieves. A kernel at 19 GB/s is leaving half the machine idle, and *which half*
(transactions too small, reuse thrashing, issue slots burned) decides the fix.
The first port's kernels ran at 15–20 GB/s; the survivors run close to the
achievable ceiling, which is why I stopped: the remaining kernels are within
~25% of the bandwidth bound, and further effort would buy single milliseconds.

## 6. Why it is fast, part II: the three failures that shaped it

**A warp-serial scan of the recurrence: 55.7 ms, worse than what it replaced.**
The recursive filter is the awkward part of the port — a per-step integer
recurrence with truncation, so the classic
[block-parallel GPU formulation (Nehab et al. 2011)][gl-nehab], which *reassociates* the
filter algebraically, cannot reproduce it
bit-exactly. My first "clever" alternative kept bit-exactness by letting the 32
lanes of a warp take turns via [shuffles][gl-warp]: coalesced, exact — and 31 of 32 lanes
burn issue slots on every serial step. Instruction throughput, not bandwidth.
The k-minor layout made the whole question moot: with a lane per *disparity*
rather than per *position*, every lane runs its own recurrence and nothing is
serial. The failed experiment is documented in the source, because it looks like
a good idea and will look like one again.

**An ordered tie-break in the top-2 reduction: ten wrong pixels in 407,040.**
The fused top-2 reduces across the warp with a `shfl_down` tree, and the tree
merges *non-adjacent* disparity ranges — lane 0 combines lane 16 before lane 1 —
so any tie rule of the form "the other side holds larger disparities" is wrong
mid-tree. The output differed from the CPU in exactly ten pixels, every one an
exact score tie. A five-million-run host simulation of the precise shuffle tree
reproduced it, and the fix packs (value, k) into a single integer whose plain
comparison is (value descending, k ascending) — order-independent, two shuffles
per round, and the simulation now lives in `make test` as a half-million-trial
guard. Without bit-identity as the referee, ten tie-broken pixels would have
passed any visual inspection and any accuracy benchmark.

**A compile-time run length: six of eight scenes silently wrong.** The k-runs
were padded to a hard-coded 64, and six of the eight ground-truth scenes need
D=80: their upper disparities were never scored, and the top-2 read into the
*neighbouring pixel's* run. The two scenes that pass — Teddy and Cones — are
exactly the two everything gets tuned on, at D=60. The identity check runs all
eight scenes precisely for this reason, and this bug is why it keeps doing so.

## 7. Verification as a design constraint

The whole port was built under one rule: **every intermediate keeps the CPU's
exact integer arithmetic, so `cmp` on the final disparity map is the test.** The
census bit order, the [Q14][gl-q14] tables with C++ truncating division, the filter's
int32-carry/int16-store pattern, the ascending-k strictly-greater top-2 — all
replicated. This cost something (the Nehab-style filter was off the table), and
it paid for itself three times in section 6: the tie bug and the padding bug are
*invisible* to accuracy benchmarks at the scale they occur, and the pipelined
mode's races would be too.

It also buys something rarely available in GPU ports: the accuracy story needs
no new evidence. The GPU produces the same bytes; Part 2's comparison against
SGM *is* the GPU's comparison against SGM.

## 8. What the GPU taught me about the CPU implementation — tested, and mostly no

I first drafted this section as three promising transfers. Then I built and
measured them the same night, and the honest version is better than the
promising one:

1. **Fuse the top-2 insert into the filter's last pass, delete the plane store**
   — the GPU's biggest single win, mapped back. Built, bit-identical, and **a
   wash**: filter+insert 337 ms thread-summed unfused against 329 fused, wall
   time equal to slightly worse. The mechanism is the point: on the GPU the
   deleted stores were DRAM traffic; on the CPU **the plane sits in L2 between
   passes** (814 KB against 2 MB), so the store the fusion deletes was nearly
   free and the re-read was nearly a cache hit. A first version was 18% worse
   outright, because putting the branchy insert inside the recurrence loop
   killed the compiler's [autovectorisation][gl-autovec] — a trap worth its own sentence:
   on a CPU, fuse loops only if the hot loop stays branch-free.
2. **Pin the worker threads to the A57 cluster** — it transformed the GPU
   tool's solve, so surely the CPU tool too. Measured: **40% worse on the
   mean, ten times tighter on the spread.** The two Denver cores are both the
   variance *and* real throughput; the CPU-only matcher needs them, while the
   GPU tool's four solve threads are better off leaving them to the CUDA
   driver. Reverted, recorded.
3. **Score fused into the first filter pass** — not built, and I will say why
   rather than imply it is pending: it shares the exact mechanism of #1 (the
   score plane is L2-resident when the horizontal pass reads it), so the same
   null result is predicted. That is an inference, not a measurement, and it
   is labeled as one.

So the transferable lesson is not a technique. It is the series' thesis
completing itself: the desktop was not a proxy for the TX2, the TX2's CPU is
not a proxy for its own GPU, and **an optimisation is a measurement attached to
a memory system, not a portable fact.** What does transfer is the method —
bit-identity as referee, all eight scenes always, and the willingness to revert
what the numbers refuse.

## 9. Where this leaves the project

The matcher now runs at the camera's full resolution faster than the camera
delivers frames, at the operating point that beats OpenCV's SGM on ground-truth
accuracy, on a computer that costs less than the camera. The margin is 13%
(28.9 vs 33.3 ms) at locked clocks; thermal throttling on a vehicle will eat
into it, and that is the honest caveat on the headline.

What it unblocks is the actual plan: the sparse [feature][gl-keypoints] path, object
tracking,
and a temporal prior — the previous frame's disparities are exactly the mask
that Part 2's [coarse-to-fine][gl-c2f] machinery wants, and unlike the half-resolution
coarse pass, they are free. The GPU also remains mostly idle in the frame
budget: ~26 ms of kernels leaves room the eventual CNN detector will claim, and
if it claims too much, this pipeline degrades gracefully to 15 Hz rather than
falling over.

**References**

Full citations with DOIs, along with every term this post uses, are in the
[series glossary][gl-appendix].

- [Ruf et al., *ReS2tAC — UAV-Borne Real-Time SGM Stereo Optimized for Embedded
  ARM and CUDA Devices*][gl-res2tac], Sensors 21(11), 2021 — disparity-in-the-lanes, the
  assignment this project has now re-derived three times.
- [Nehab, Maximo, Lima, Hoppe, *GPU-Efficient Recursive Filtering and Summed-Area
  Tables*][gl-nehab], SIGGRAPH Asia 2011 — the block-parallel recursive filter, and by its
  existence, the explanation of why warp-serial scans lose.
- [Bleyer, Rhemann, Rother, *PatchMatch Stereo*][gl-patchmatch], BMVC 2011;
  [Geiger et al., *ELAS*][gl-elas], ACCV 2010 — the avoid-the-sweep family Part 2
  measured against this design.
- Parts [1](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/)
  and [2](https://www.mariolueder.com/2026-08-08-Dense-MASDA-Belief-Propagation-Stereo-on-a-Jetson/)
  of this series — the derivation of MASDA, and the dense matcher this post
  makes real-time. The [glossary of terms and sources][gl-appendix] serves all
  three.

[gl-appendix]: https://www.mariolueder.com/masda-glossary/
[gl-masda]: https://www.mariolueder.com/masda-glossary/#masda
[gl-lbp]: https://www.mariolueder.com/masda-glossary/#loopy-belief-propagation
[gl-factorgraph]: https://www.mariolueder.com/masda-glossary/#factor-graph
[gl-messages]: https://www.mariolueder.com/masda-glossary/#messages-responsibility-and-availability
[gl-greedy]: https://www.mariolueder.com/masda-glossary/#greedy-decode
[gl-one2one]: https://www.mariolueder.com/masda-glossary/#one-to-one-constraint
[gl-margin]: https://www.mariolueder.com/masda-glossary/#margin-and-the-margin-gate
[gl-keypoints]: https://www.mariolueder.com/masda-glossary/#keypoints-and-detector-repeatability
[gl-disparity]: https://www.mariolueder.com/masda-glossary/#disparity
[gl-costvolume]: https://www.mariolueder.com/masda-glossary/#cost-volume
[gl-census]: https://www.mariolueder.com/masda-glossary/#census-transform
[gl-tad]: https://www.mariolueder.com/masda-glossary/#truncated-absolute-difference
[gl-rf]: https://www.mariolueder.com/masda-glossary/#edge-aware-recursive-filter
[gl-sgm]: https://www.mariolueder.com/masda-glossary/#semi-global-matching
[gl-c2f]: https://www.mariolueder.com/masda-glossary/#coarse-to-fine
[gl-layout]: https://www.mariolueder.com/masda-glossary/#disparity-major-and-disparity-minor-layout
[gl-metrics]: https://www.mariolueder.com/masda-glossary/#coverage-precision-and-the-bad-pixel-rate
[gl-bestofn]: https://www.mariolueder.com/masda-glossary/#interleaved-best-of-n
[gl-patchmatch]: https://www.mariolueder.com/masda-glossary/#patchmatch-stereo
[gl-elas]: https://www.mariolueder.com/masda-glossary/#elas
[gl-res2tac]: https://www.mariolueder.com/masda-glossary/#res2tac
[gl-nehab]: https://www.mariolueder.com/masda-glossary/#gpu-efficient-recursive-filtering
[gl-tx2]: https://www.mariolueder.com/masda-glossary/#jetson-tx2
[gl-cores]: https://www.mariolueder.com/masda-glossary/#a57-and-denver-cores
[gl-q14]: https://www.mariolueder.com/masda-glossary/#q14-fixed-point
[gl-warp]: https://www.mariolueder.com/masda-glossary/#warp-coalescing-and-shuffle
[gl-cmem]: https://www.mariolueder.com/masda-glossary/#constant-and-shared-memory
[gl-pinned]: https://www.mariolueder.com/masda-glossary/#pinned-memory-and-io-coherency
[gl-fusion]: https://www.mariolueder.com/masda-glossary/#kernel-fusion
[gl-bandwidth]: https://www.mariolueder.com/masda-glossary/#bandwidth-bound
[gl-autovec]: https://www.mariolueder.com/masda-glossary/#autovectorisation
[gl-bitid]: https://www.mariolueder.com/masda-glossary/#bit-identity
