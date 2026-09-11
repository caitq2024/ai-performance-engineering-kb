# AI Performance Engineering Knowledge Base Notes

Chinese study materials derived from *AI Systems Performance Engineering*.

## Index

### Chapter 5: GPU-Based Storage I/O Optimizations

Directory: `notes/chapter-05-gpu-storage-io/`

- `chapter05-presentation.pdf` + `chapter05-presentation.tex`: 31-page Beamer deck (same style as the chapter 6 v3 deck) — five parts: storage fundamentals (locality, sequential vs. random reads, NVMe/kernel tuning), direct GPU data paths (GDS, cuFile, gdsio before/after, cuda-checkpoint, DeepSeek 3FS), shared storage at scale (NFS, Lustre striping, S3 staging, replicate/compress, monitoring), the data pipeline (DataLoader knobs, pinned-memory/stream overlap, DALI, NeMo Curator), and the continuous profiling loop incl. the batch-size experiment for comm- vs. compute-bound diagnosis.
- `chapter05-lecture-script-zh.md`: slide-by-slide Chinese lecture script (31 pages / ~50 minutes) with 🎤 spoken content, 大白话 one-liners, ❓ anticipated Q&A, time budget, and overrun fallbacks.
- `chapter05.pdf`: extracted source chapter PDF (28 pages).
- `figs/fig5-*.png`: 2 supporting figures (VAST with/without GDS architecture; 3FS components).

### Chapter 6: GPU Architecture, CUDA Programming, and Occupancy

Directory: `notes/chapter-06-gpu-cuda-occupancy/`

- `chapter06-presentation-v3.pdf` + `chapter06-presentation-v3.tex` + `chapter06-lecture-script-v3-zh.md`: **integrated final** — Parts I–V merged, single title page, 56 pages; script keeps first-person spoken content only; adds an arithmetic-intensity-by-example page after the roofline plus three memory-module worked-example pages (p.33 SMEM-vs-L1 sliding filter, p.35 TMEM/TMA accumulate-and-prefetch, p.37 constant-cache broadcast); cuts __launch_bounds__, Occupancy API, and Compute Sanitizer for time (kept in the part drafts).
- `chapter06-part1-draft.pdf` + `chapter06-part1-script-draft-zh.md`: restructured Part I preview (21 pages) — SM internals → factory/workshop analogies → threads/warps/blocks/grids → SIMT meaning → divergence → worked example → occupancy/limits/PTX; script covers every slide with 🎤/📖/大白话/❓ blocks.
- `chapter06-part2-draft.pdf` + `chapter06-part2-script-draft-zh.md`: Part II preview (9 pages) — CUDA Programming Refresher: kernel anatomy, six-step host flow, why-pass-N, bounds check, launch-parameter recipe (slimmed), 2D/3D, async allocation and memory pools.
- `chapter06-part3-draft.pdf` + `chapter06-part3-script-draft-zh.md`: Part III preview (10 pages) — memory ladder, registers/spilling, shared+L1, TMEM/TMA, constant cache, L2, HBM3e/dual-die, Unified Memory and its taming.
- `chapter06-part4-draft.pdf` + `chapter06-part4-script-draft-zh.md`: Part IV preview (10 pages) — occupancy ground rules, addSequential/PyTorch-loop traps, addParallel, nsys/ncu, the 22x-at-38.7%-occupancy verdict, memory-bound reality check (LLM decode), __launch_bounds__, occupancy API.
- `chapter06-part5-draft.pdf` + `chapter06-part5-script-draft-zh.md`: Part V preview (8 pages) — Compute Sanitizer, roofline analysis, lower-precision bandwidth wins, profiling workflow, key takeaways, conclusion, references.
- `chapter06-notes-zh.md`: chapter reading notes on GPU architecture, CUDA, memory hierarchy, occupancy, profiling, and roofline analysis.
- `chapter06-notes-full-zh.md`: expanded Chinese chapter notes aligned with the full source chapter.
- `chapter06-lecture-script-zh.md`: bilingual, slide-by-slide lecture script for the 52-page chapter presentation.
- `chapter06-lecture-script-v2-zh.md`: revised 53-page bilingual lecture script matching the v2 deck (Part I follows the book's section order; factory/workshop narrative, part transitions).
- `chapter06-presentation.tex`: editable Beamer presentation source (52 pages).
- `chapter06-presentation.pdf`: rendered 52-page presentation.
- `chapter06-presentation-v2.pdf`: revised 53-page presentation aligned with the v2 lecture script; Part I restored to the book's section order, worked-example and analogy pages at 7/9/14/15.
- `chapter06-presentation-v2.tex`: editable Beamer source for the v2 presentation.
- `chapter06-presentation-zh.pdf`: fully Chinese-localized deck (完全汉化版), page-for-page mirror of the 52-page presentation.
- `chapter06-presentation-zh.tex`: source for the Chinese deck; compile with XeLaTeX (uses xeCJK + Noto Sans CJK SC).
- `chapter06.pdf`: extracted source chapter PDF (48 pages).
- `figs/fig6-*.png`: 22 supporting figures referenced by the presentations.
- `figs/analogy-*.png`: 2 presenter-made analogy figures (SM workshop, GPU factory) used by the v1/v2 decks.

### Chapter 12: Dynamic Scheduling, CUDA Graphs, and Device-Side Orchestration

Directory: `notes/chapter-12-dynamic-scheduling-cuda-graphs/`

- `chapter12-notes-zh.md`: chapter reading notes on dynamic work queues, CUDA Graphs, device-side launch, and multi-GPU orchestration.
- `chapter12-presentation.tex`: editable Beamer presentation source.
- `chapter12-presentation.pdf`: rendered presentation.
- `chapter12.pdf`: extracted source chapter PDF (46 pages).
- `figs/fig12-*.png`: 17 supporting figures referenced by the presentation.

## Scope

The collection intentionally excludes the full book PDF and Chapter 1/2 presentations. It includes the extracted Chapter 6 and Chapter 12 source PDFs alongside their matching study materials.
