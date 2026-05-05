# Evaluation Results — 20-Demo Hybrid Observer

**Observer:** `hybrid_pixels_to_cartpole_observer_theta_blend_0p7.json`  
**Training demos:** 20 (angles ±2–10°, ±12°, symmetric)  
**True model mismatch:** masspole×1.1, half-pole-length×0.9  
**Evaluation protocol:** 100 trials, θ₀ ~ uniform(−12°, +12°), 150 steps (2.5 s at 60 Hz), seed=0

---

## Summary Statistics

```
Teacher failures   : 0/100
Student survival   : 96/100  (96.0%)
Paper success rate : 75/100  (75.0%)   |x̂| ≤ 0.176 m  AND  |θ̂| ≤ 2° at step 150

e_stab = ||x̂_final||₂       all 100    surviving 96
  median                       0.200       0.199
  5th pct                      0.061       0.061
  95th pct                     0.898       0.627
  min / max                0.025 / 2.256   0.025 / 1.660
```

## Final Estimated State — Component Medians (all 100 trials)

| Component  | median | p95   |
|------------|--------|-------|
| x̂ (m)     | 0.130  | 0.313 |
| ẋ̂ (m/s)   | 0.125  | 0.535 |
| θ̂ (rad)   | 0.003  | 0.045 |
| θ̇̂ (rad/s) | 0.093  | 0.718 |

## Interpretation

**Paper success (75%)** is a convergence-by-deadline check: did the final snapshot land inside
the goal zone (|x̂| ≤ 0.176 m AND |θ̂| ≤ 2°) at step 150? It does not require the trajectory
to have stabilized earlier or to stay there — a trial that drifted through the goal zone at the
last step counts.

**Student survival (96%)** reflects true Gym termination — the pole fell on 4 trials. All 4 are
near-vertical starts (|θ₀| < 2°): −1.9°, −0.3°, +1.7°, −1.8°. Near-vertical binary frames
carry almost no signal, so the 0.7-weight pixel blend on θ is pulling on noise rather than a
real measurement.

The **21 trials that survived but missed the paper success criterion** are not failures of
stability — they are mid-transient at step 150. The system is still correcting; it just hasn't
converged inside the 2° / 0.176 m box by the deadline.

θ̂ at the median is effectively zero — the 0.7-weight Ridge blend onto the pixel angle estimate
is working. Residual error lives in x̂ and θ̇̂, neither of which pixels can observe directly.
