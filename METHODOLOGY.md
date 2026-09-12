# Methodology

The rules every project in this series follows. They exist because each one is a
mistake I made once and would rather not make again.

## Working

1. **One change at a time, sized to a testable end state.** A change that cannot
   be verified on its own gets split until it can be.
## Correctness

2. **A reference implementation exists before the optimised one.** Slow, obvious,
   high precision, no tricks. Everything is measured against it. Write it first
   even when it feels like a detour.
3. **Never loosen a tolerance to make a test pass.** If the fast path disagrees
   with the reference beyond the bar, the fast path is wrong. Fix it.
4. **Relative tolerance beats absolute.** The bar is "no worse than the naive
   implementation in the same precision", not a magic epsilon. Absolute
   thresholds either pass broken code or fail correct code.
5. **Test the boundaries.** Non power of two sizes, prime lengths, size 1, empty
   inputs, and the largest thing that fits. Bugs live at the edges.

## Measurement

6. **No number that did not come from a measurement.** Not from the paper, not
   estimated from the algorithm, not "roughly". Measured or absent. This rule is
   the reason any of these results are worth reading.
7. **Report medians, with the range.** One scheduling spike ruins a mean.
8. **Warm up, then synchronise, then time.** Accelerator calls are async. Every
   benchmark that looks impossibly fast is measuring launch overhead.
9. **Fix seeds and record them.** A result you cannot reproduce is a result you
   do not have.
10. **Report variance, not just the point estimate.** Three seeds minimum for
    anything involving training. A single seed improvement of two percent is
    noise, and it will be read as noise.
11. **A metric gets a test before any method does.** The metric is the instrument.
    A broken one makes every downstream number wrong in a way that looks like a
    modelling problem, which is where the time goes.

## Recording

12. **Negative results stay in.** "I tried X, it was worse, here is the evidence"
    is signal. A results table containing only wins is not believable.
13. **Say "not measured on this hardware"** rather than extrapolating. Nobody is
    fooled by a number for a device you do not own.
14. **Every experiment that moves a number gets a logbook entry.** Three lines:
    tried, measured, concluded. Write the concluded line even when it is "no idea
    why", because those are the ones worth returning to.
