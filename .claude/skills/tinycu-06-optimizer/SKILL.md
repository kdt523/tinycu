---
name: tinycu-06-optimizer
description: Phase 6 of TinyCU — pass manager and -O1 optimizations on the IR (constant folding/propagation, algebraic simplification, copy propagation, dead code elimination, CFG simplification) with per-pass dumps and measured impact.
---

# Phase 6 — Optimizer (`-O1`)

## Goal
A small, correct, measurable optimizer. Correctness > aggressiveness: every pass must keep
interpreter output identical (checked in phase 7 fuzzing).

## 1. Pass manager (`src/opt/passmanager.cpp`)
- `struct Pass { virtual const char* name(); virtual bool run(Function&); }` (returns changed).
- Pipeline `-O1`: `constfold → copyprop → simplifycfg → dce`, iterate to fixpoint (max 10).
- `-O0`: no passes. `--passes=constfold,dce` custom list. `--dump-after=<pass>` prints IR.
- `--verify-each` runs verifier after each pass. `--stats` prints per-pass change counters.
- `--disable-pass=<name>` (used by the reducer/bisection in phase 7).

## 2. Passes
1. **Constant folding + local constant propagation** (per basic block): track vreg→constant
   map; fold arith/cmp/cvt/min/max/abs with **exact TinyCU semantics** (i32 wraparound via
   uint32_t; never fold div/rem by zero or INT_MIN/-1 — leave the instruction; f32 folding
   uses `float` math identical to the interpreter; never fold sqrt of negative).
   `condbr` on constant → `br`.
2. **Algebraic simplification**: `x+0, x-0, x*1, x/1 → x`; `x*0 → 0` **int only**
   (float: NaN/−0 issues — document why in a comment); `x-x → 0` int only; `!!p → p`.
3. **Copy propagation** (per block): after `mov a, b`, replace uses of `a` with `b` until
   `a` or `b` is redefined.
4. **Dead code elimination**: remove side-effect-free instructions whose dest is never used
   anywhere in the function (iterate). Loads are removable; stores/barrier/terminators never.
   Upgrade (optional): liveness-based DCE using backward dataflow over the CFG.
5. **SimplifyCFG**: remove unreachable blocks; merge block with single successor whose only
   predecessor it is; thread `br` to empty block containing only `br`; turn `condbr p, L, L` → `br L`.

## 3. Measurement (feeds resume numbers)
`python -m tcharness.optstats` over all kernels: PTX instruction count, `.reg` count and
ptxas `Used N registers` at -O0 vs -O1 → results/opt_impact.md. Note honestly: ptxas itself
optimizes, so SASS-level gains may be small — report both PTX and SASS deltas and explain.

## Tests
- Unit tests per pass on hand-written IR snippets (before/after text), ≥8 per pass incl.
  edge cases: wrap overflow fold, division by zero not folded, float `x*0` not folded,
  copy-prop across redefinition, DCE keeps stores.
- Golden `--dump-after=` outputs for 5 programs.
- All phase-5 kernels: `-O1` output == `-O0` output (interpreter + GPU).

## Definition of Done
- [ ] `-O1` pipeline passes verifier on everything; all kernel results unchanged
- [ ] results/opt_impact.md generated
- [ ] Tag `v0.6`

## Interview notes
Local vs global optimization; dataflow analysis (liveness = backward, reaching defs =
forward), lattice/fixpoint idea; why folding must match target semantics exactly (classic
miscompile source); why float algebra is not associative; register pressure vs optimization.
