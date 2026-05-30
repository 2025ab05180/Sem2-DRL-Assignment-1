# Team 108 - MAB Report Notes

## Dataset Design

The group number is `G = 108`. Therefore the number of medicines is `K = (108 mod 3) + 5 = 5`. The hidden success probabilities are calculated using `P_i = 0.4 + ((G + i) mod 6) * 0.07`, giving medicines `0` to `4` probabilities `0.40`, `0.47`, `0.54`, `0.61`, and `0.68`. Medicine `4` is therefore the true best medicine.

The patient dataset contains exactly `1000` patients. Each patient has `patient_id` from `0` to `999` and severity score `(patient_id mod 5) + 1`. During every algorithm run, the script dynamically fills `assigned_medicine`, `clinical_outcome`, and `utility_score`.

## Strategy Results

| Strategy | Final cumulative reward | Final best medicine | True best medicine selected | Convergence patient | Stability score |
|---|---:|---:|---:|---:|---:|
| Immediate Exploitation | 466.7 | 4 | 960 | 43 | 2.436 |
| Epsilon-Greedy 1% | 480.5 | 4 | 980 | 14 | 2.383 |
| Epsilon-Greedy 10% | 422.0 | 4 | 179 | 849 | 2.166 |
| Epsilon-Greedy 50% | 424.4 | 4 | 575 | 58 | 2.329 |
| UCB1 Confidence-Based | 431.2 | 4 | 442 | 109 | 2.973 |

## Comparative Answers

1. The highest cumulative reward at the end of 1000 patients is achieved by `Epsilon-Greedy 1%` with reward `480.5` in this seeded run.
2. The fastest stable identification of the true best medicine is also `Epsilon-Greedy 1%`, which converges at patient `14`.
3. The most stable cumulative performance is `Epsilon-Greedy 10%` based on the lowest rolling reward variation score.
4. The safest real-world recommendation is `UCB1`, because it gives under-tested medicines structured confidence-based opportunities early and reduces exploration naturally as evidence grows.

## Short Summary

Immediate exploitation performs well here because the initial testing phase correctly identifies medicine `4`, but this strategy is risky because early random outcomes can lock future patients into a wrong treatment. Epsilon-greedy with `1%` exploration gives the highest reward in this run, while `50%` exploration tests too much and loses treatment utility. UCB1 is a safer hospital deployment strategy because it balances patient benefit with evidence gathering in a controlled way.
