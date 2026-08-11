---
layout: post
title: 'Real-Time Dense Stereo on a Jetson TX2 (MASDA, Part 3)'
subtitle: 'The GPU takes the image plane, the CPU keeps the graph: kernel dataflow, the memory layout the CPU rejected three times, bit-identity as the referee, and what the board delivers at each resolution.'
thumbnail-img: /assets/img/2026-08-09-Realtime-Dense-MASDA_files/real_pair.png
date: '2026-08-09 01:00:00 +0200'
categories: association
comments: false
mathjax: true
author: Mario Lüder
tags: [belief-propagation, data-association, computer-vision, embedded, cuda]
---

This is the real-time implementation: [dense MASDA][p2] running on a
[Jetson TX2][gl-tx2] at **31.7 ms per frame at 848×480** — the camera's full
resolution, faster than the camera delivers frames — with the disparity map
[bit-identical][gl-bitid] to the CPU matcher on all eight ground-truth scenes.

The design question is not "which parts go on the GPU". It is that the matcher has two
halves with opposite computational shapes, and the TX2 has two processors with
opposite strengths. Getting that mapping right is most of the work; the rest is memory
layout, and the memory layout is where the surprise is.

*Every term is defined in the [series glossary][gl-appendix]. [Part 1][p1] derives the
message equations; [Part 2][p2] is the matcher this post makes real-time.*

Here is what it produces on the camera it is built for — the [D435 IR pair][gl-d435],
with the structured-light projector on and off:

![real pair](/assets/img/2026-08-09-Realtime-Dense-MASDA_files/real_pair.png)

The projector is the reason this project uses MASDA at all. With it on, the scene is
covered in dots that make texture where there was none, and the matcher answers 88% of
the image. With it off, the same scene, the same code: 78%, and the surfaces break up
into speckle wherever the room is flat and grey. Those dots also make the descriptors
**3.3× degenerate** — many pixels have identical Census codes — which is exactly the
ambiguity a [one-to-one constraint][gl-one2one] is for, and exactly where a
winner-take-all matcher assigns the same right pixel to several left ones.

---

## 1. The split: regular work on the GPU, irregular work on the CPU

The two halves of the matcher differ in kind:

- **The cost side is regular.** Census, the graded cost, the [recursive
  filter][gl-rf] and the top-2 selection touch every (pixel, disparity) pair in a
  pattern known at compile time. No data-dependent branching, no irregular memory.
- **The solve side is irregular.** [MASDA's][gl-masda] [messages][gl-messages] and its
  decode walk claimant lists per right pixel
  and applies a [greedy][gl-greedy] one-to-one assignment over a sorted order. It is
  also *small*: with two candidates per pixel the whole row problem fits in 28 KB of
  cache, and four ARM cores handle a full frame in ~11 ms.

Making the solve a GPU kernel would be work spent making the architecture uniform, not
work spent making the system faster. So the interface is **two scored disparity
candidates per pixel** — 8 MB per frame instead of a 49 MB [cost volume][gl-costvolume].
The GPU reduces; the CPU decides.

```
   GPU (256-core Pascal)                     CPU (4× A57, pinned)
  ┌────────────────────────────────┐        ┌──────────────────────────┐
  │ census L,R    (48 bits/px)     │        │  MASDA row decode        │
  │ filter coeffs (LUT per px)     │        │  one-to-one greedy claim │
  │ graded cost   (Census + AD)    │        │  margin gate             │
  │ edge-aware recursive filter    │        │  sub-pixel parabola fit  │
  │ running top-2 per pixel        │        │                          │
  └───────────────┬────────────────┘        └────────────▲─────────────┘
                  │  2 × (score, d) + count + 2 neighbours per pixel
                  └──────────────  8 MB / frame  ────────┘
```

One Tegra-specific trap lives on that arrow and cost 300 ms twice before I believed it:
**the TX2 has no I/O coherency, so every flavour of [`cudaHostAlloc`][gl-pinned] memory
— including the one the documentation's mental model calls "cached pinned" — is
uncached on the CPU side.** A solver reading candidates from such memory runs about
seven times slower than from ordinary pageable memory. The candidates travel through a
staged `cudaMemcpy` into a plain `std::vector`, and section 3 shows where that copy
hides.

## 2. Dataflow: five kernels, two of which are fusions

Per frame, in stream order, with measured kernel times at 848×480:

```
 frame ──► upload L,R                                        (~1 ms)
        ──► k_census        L,R → 48-bit descriptors          (1.5 ms
        ──► k_rf_coeffs     |∇I| → filter coefficients          with coeffs)
        ──► k_score_hfwd    census → score → horizontal        (8.8 ms)
        │                   FORWARD filter pass, FUSED:
        │                   the raw score never touches DRAM
        ──► k_hbwd          horizontal backward pass           (5.0 ms)
        ──► k_vert_fwd      vertical forward pass              (3.3 ms)
        ──► k_vert_bwd_top2 vertical backward pass with the    (7.4 ms)
        │                   TOP-2 FUSED in as a warp
        │                   reduction; its own stores are
        │                   deleted — nothing reads them
        ──► fetch           candidates → cached host memory    (hidden, §3)
        ──► CPU decode      480 independent rows on 4 A57s     (11 ms, parallel)
```

The two fusions are where most of the speed lives, and they follow one principle:

> **A pass that stores exactly what the next pass reads is a fusion candidate. A store
> that nothing reads afterwards is a bug you are paying for.**

The first port scored the volume (write 47 MB), read it back to filter (94 MB), wrote
it filtered, then read it again for the top-2. The shipping version computes the score
*inside* the first filter pass and consumes the volume *inside* the last one. The
scored-but-unfiltered volume never makes a round trip and the fully-filtered volume is
never materialised at all. What remains in DRAM is the minimum the data dependencies
allow — the recurrences genuinely need their intermediate planes.

**The sub-pixel fit rides along for free here.** [Part 2][p2] describes the fit and what
it is worth: 41.5% → 25.2% bad-1.0. On the CPU it costs 1.30×, because the two
neighbouring costs have to be retained while streaming planes. On the GPU they are
already in registers: the top-2 reduction has the whole disparity range of a pixel live
across the warp at the moment the winner is known, so publishing the winner takes one
broadcast and fetching its neighbours takes two shuffles. Measured cost: **1.10×**, and
sending those neighbours to the host in Q14 rather than float halves the extra transfer.

## 3. The pipeline, and where detection hides

Single-frame latency is ~52 ms. Throughput is 31.7 ms because the GPU computes frame
$$t{+}1$$ while the CPU works on frame $$t$$:

![overlap](/assets/img/2026-08-09-Realtime-Dense-MASDA_files/overlap.png)

Three details make the overlap real rather than hopeful.

**The fetch runs on its own thread.** `cudaMemcpy` serializes with the stream, so the
fetcher blocks until frame $$t{+}1$$'s kernels finish and then copies — while the
decode of frame $$t$$ is still on the A57s. Before this, the copy sat serially in the
loop.

**The decode threads pin themselves to the A57 cluster.** The TX2 has [four A57s and
two Denver cores][gl-cores]; unpinned, the scheduler wanders across both and the decode
varied 30–45 ms run to run. Pinned, it sits within half a millisecond of its minimum.
The Denvers are left to the CUDA driver and the fetcher. (The same change measured *40%
worse* on the CPU-only matcher, which needs the Denvers for throughput. An optimisation
is a measurement attached to a machine.)

**Keypoint detection is a third thread, and it is free.** The system needs a sparse
feature set as well as a dense map — for tracking and odometry — and detection is 29 ms
of one core. Run inside the pipelined loop beside the decode, it costs **0.4–1.4 ms of
frame time**, because it hides under the 26 ms of kernels. 97.3% of detected keypoints
carry a disparity read straight out of the dense map.

That last point is a result, not plumbing. Sampling the dense map at the keypoints
beats running the sparse matcher on the same keypoints on every axis: **0.853 precision
against 0.706, with 57% more correct matches**, and no matcher to run. The sparse
matcher's recall is bounded by whether the *right* image's detector also fired within a
pixel of the true correspondence — a 44–51% repeatability ceiling — and a dense map has
no such requirement. One producer, two products.

It had to be measured *inside* the loop to say any of this. Bolted on after it, the
detection step timed nothing and the frame rate looked unchanged for the wrong reason.

## 4. The layout that ends the `[d][x]`-versus-`[x][d]` question

Parts 1 and 2 kept running into the same question: is the cost volume
[disparity-major or disparity-minor][gl-layout]? The CPU answered three times —
disparity-major, because the aggregation filter wants whole constant-disparity planes,
and the transpose to the other layout was the dominant memory cost of the early
implementation.

**The GPU wants the opposite, and the reason is precise:**

![layout](/assets/img/2026-08-09-Realtime-Dense-MASDA_files/layout.png)

The volume is stored `vol[y][x][k]` — $$k$$ innermost, padded to 64-aligned runs — and
a [warp][gl-warp] is **32 consecutive disparities of one image row**:

- The right-census reads for 32 consecutive $$d$$ at one $$x$$ are 32 *consecutive*
  addresses: one coalesced 256-byte window that slides one element per step and lives
  in L1. The left descriptor and the filter coefficient are the same address for all
  32 [lanes][gl-lanes] — a hardware broadcast.
- Every volume access anywhere in the pipeline is a $$k$$-run: an aligned 64- or
  128-byte transaction. Nothing strides.
- Each lane carries its own filter recurrence in registers. No lane waits on another,
  and the filter needs no shuffles at all.

The same assignment appears in [ReS2tAC][gl-res2tac] for SGM, on NEON as well as CUDA,
and this is the third time this project has re-derived their design point from a
different direction. **Neither layout was wrong; each machine's memory system picks its
own.** A cache hierarchy with 512 KB of shared L2 wants one plane at a time, resident.
A latency-hiding machine with 32-wide transactions wants the innermost index to be the
one the warp spans. Trying to make one implementation serve both was the actual mistake.

## 5. What the board delivers

Measured steady state, pipelined over 30 frames, best of three at locked clocks:

![rates](/assets/img/2026-08-09-Realtime-Dense-MASDA_files/rates.png)

| resolution | disparities | ms/frame | rate |
|---|---|---|---|
| 424×240 | 60 | 8.7 | 115 Hz |
| 450×375 | 60 | 13.7 | 73 Hz |
| 640×480 | 60 | 24.1 | 42 Hz |
| **848×480** (sensor native) | **60** | **31.7** | **31.5 Hz** |
| 848×480 | 96 | 48.6 | 20.6 Hz |
| 848×480 | 128 | 48.4 | 20.7 Hz |

Two practical notes. **Disparity range costs in steps of 64**, because the $$k$$-runs
pad to a multiple of 64 — so $$D=128$$ is free if you are already paying for $$D=96$$,
and staying at $$D \le 64$$ halves the volume. And **848×480 at $$D=60$$ is the only
configuration that closes 30 Hz**, at 96% of the frame budget. Thermal throttling on a
vehicle will eat into that margin; the pipeline degrades to 15 Hz rather than falling
over.

The 848×480 rows use a synthetic pair — two consecutive frames from one channel —
because the recorded bags hold no rectified 848×480 left/right dump. The kernels sweep
a fixed $$D$$ and are data-independent, so their timing carries; the CPU decode is
mildly data-dependent, which is why this reads 31.7 against 31.9 on a real pair.

## 6. What did not work

**A warp-serial scan of the recurrence: 55.7 ms, worse than what it replaced.** The
recursive filter is the awkward part of the port — a per-step integer recurrence with
truncation, so the classic [block-parallel formulation (Nehab et al.)][gl-nehab], which
reassociates the filter algebraically, cannot reproduce it bit-exactly. My first
alternative kept bit-exactness by letting the 32 lanes take turns via shuffles:
coalesced, exact, and 31 of 32 lanes burning issue slots on every serial step.
Instruction throughput, not bandwidth. The $$k$$-minor layout made the question moot —
a lane per *disparity* rather than per *position* means every lane runs its own
recurrence.

**An ordered tie-break in the top-2 reduction: ten wrong pixels in 407,040.** The fused
top-2 reduces across the warp with a `shfl_down` tree, and the tree merges
*non-adjacent* disparity ranges — lane 0 combines lane 16 before lane 1 — so any tie
rule of the form "the other side holds larger disparities" is wrong mid-tree. Every one
of the ten differing pixels was an exact score tie. A five-million-run host simulation
of the precise shuffle tree reproduced it; the fix packs (value, $$k$$) into a single
integer whose plain comparison is (value descending, $$k$$ ascending), which is
order-independent. The simulation now runs in `make test`.

**A compile-time run length: six of eight scenes silently wrong.** The $$k$$-runs were
padded to a hard-coded 64, and six of the eight ground-truth scenes need $$D=80$$: their
upper disparities were never scored, and the top-2 read into the *neighbouring pixel's*
run. The two scenes that passed — Teddy and Cones — are exactly the two everything gets
tuned on, at $$D=60$$.

**Mapping the GPU's wins back to the CPU: mostly no.** Fusing the top-2 insert into the
filter's last pass is the GPU's single biggest win. On the CPU it is a wash — the plane
sits in L2 between passes, so the store the fusion deletes was nearly free and the
re-read was nearly a cache hit. A first version was 18% *worse*, because putting a
branchy insert inside the recurrence loop killed the compiler's autovectorisation.

## 7. Bit-identity as the referee

The whole port was built under one rule: **every intermediate keeps the CPU's exact
integer arithmetic, so `cmp` on the final disparity map is the test.** The census bit
order, the [Q14][gl-q14] tables with C++ truncating division, the filter's
int32-carry/int16-store pattern, the ascending-$$k$$ strictly-greater top-2 — all
replicated.

This costs something. The Nehab-style filter was off the table, and so is any
reassociation. It has paid for itself three times: the tie bug and the padding bug are
*invisible* to an accuracy benchmark at the scale they occur, and a race in the
pipelined mode would be too. The pipelined path writes the **last** frame's output, so
a broken overlap breaks the identity check instead of passing quietly.

It also means the accuracy story needs no new evidence. The GPU produces the same
bytes, so Part 2's measurements are the GPU's measurements — including the ones that
went against expectation, like the [sub-pixel fit][gl-subpix] being worth fifteen points
and the message passing not paying for itself in the dense path.

## 8. Where this leaves the project

The matcher runs at the camera's full resolution faster than the camera delivers
frames, on a computer that costs less than the camera, and it hands the rest of the
system both a dense disparity map and a sparse feature set with better disparities than
a dedicated sparse matcher produced.

What is genuinely unfinished:

- **Coverage.** 80% of pixels answered at the shipping gate against [SGM's][gl-sgm] 90%, and
  Part 2's [curve][gl-metrics] shows SGM's whole curve sitting below this one where they overlap. No
  parameter closes that; the levers that would are structural and were measured and
  declined.
- **Occlusion.** Pixels within 8 px of a depth discontinuity carry 28.6% of all error,
  and neither a sharper edge-aware filter nor a wider candidate set moves them. What is
  left is half-occlusion — support that has no counterpart in the other image — which
  is a different problem from the one the aggregation solves.
- **The temporal direction, which is where MASDA's own claim is still open.** Part 2's
  ablation says the message passing does not earn its keep deciding between two
  candidates on a rectified row. Frame-to-frame association is the opposite situation:
  the candidate set is large, two-dimensional, and genuinely ambiguous, and there is no
  dense map of the *motion* to read instead. That is the experiment the series has been
  building toward, and it has not been run yet.

**References**

Full citations with DOIs, and every term this post uses, are in the
[series glossary][gl-appendix].

- [Ruf et al., *ReS2tAC — UAV-Borne Real-Time SGM Stereo Optimized for Embedded ARM and
  CUDA Devices*][gl-res2tac], Sensors 21(11), 2021 — disparity-in-the-lanes, and the
  embedded CUDA baseline this design is measured against.
- [Nehab, Maximo, Lima, Hoppe, *GPU-Efficient Recursive Filtering and Summed-Area
  Tables*][gl-nehab], SIGGRAPH Asia 2011 — the block-parallel recursive filter, and by
  its existence the explanation of why a warp-serial scan loses.
- Parts [1][p1] and [2][p2] of this series — the derivation, and the matcher this post
  makes real-time.

[p1]: https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/
[p2]: https://www.mariolueder.com/2026-08-08-Dense-MASDA-Belief-Propagation-Stereo-on-a-Jetson/
[gl-appendix]: https://www.mariolueder.com/masda-glossary/
[gl-masda]: https://www.mariolueder.com/masda-glossary/#masda
[gl-messages]: https://www.mariolueder.com/masda-glossary/#messages-responsibility-and-availability
[gl-one2one]: https://www.mariolueder.com/masda-glossary/#one-to-one-constraint
[gl-greedy]: https://www.mariolueder.com/masda-glossary/#greedy-decode
[gl-costvolume]: https://www.mariolueder.com/masda-glossary/#cost-volume
[gl-rf]: https://www.mariolueder.com/masda-glossary/#edge-aware-recursive-filter
[gl-subpix]: https://www.mariolueder.com/masda-glossary/#sub-pixel-disparity
[gl-sgm]: https://www.mariolueder.com/masda-glossary/#semi-global-matching
[gl-metrics]: https://www.mariolueder.com/masda-glossary/#coverage-precision-and-the-bad-pixel-rate
[gl-layout]: https://www.mariolueder.com/masda-glossary/#disparity-major-and-disparity-minor-layout
[gl-res2tac]: https://www.mariolueder.com/masda-glossary/#res2tac
[gl-nehab]: https://www.mariolueder.com/masda-glossary/#gpu-efficient-recursive-filtering
[gl-tx2]: https://www.mariolueder.com/masda-glossary/#jetson-tx2
[gl-cores]: https://www.mariolueder.com/masda-glossary/#a57-and-denver-cores
[gl-d435]: https://www.mariolueder.com/masda-glossary/#realsense-d435-and-the-ir-pair
[gl-lanes]: https://www.mariolueder.com/masda-glossary/#vector-lanes
[gl-warp]: https://www.mariolueder.com/masda-glossary/#warp-coalescing-and-shuffle
[gl-q14]: https://www.mariolueder.com/masda-glossary/#q14-fixed-point
[gl-pinned]: https://www.mariolueder.com/masda-glossary/#pinned-memory-and-io-coherency
[gl-bitid]: https://www.mariolueder.com/masda-glossary/#bit-identity
