# Team 108 - DP Report Notes

## Environment Configuration

The group number is `108`, so the last digit is `8`. Since the last digit is between `5` and `9`, the environment uses a `6x6` grid. Since `8` is even, the maximum battery level is `10`. The wind disturbance probability is `30%`, and the maximum episode length is `75` steps.

The start position `S` is fixed at the top-left corner `(0, 0)`. The group-specific layout used in the code is:

| Cell type | Locations |
|---|---|
| Start `S` | `(0, 0)` |
| Rescue targets `R` | `(1, 5)`, `(3, 3)`, `(5, 4)` |
| Charging stations `C` | `(0, 4)`, `(4, 1)` |
| Danger zones `D` | `(1, 2)`, `(2, 5)`, `(4, 3)`, `(5, 1)` |
| Blocked cells `X` | `(0, 2)`, `(2, 2)`, `(3, 0)` |
| Wind zones `W` | `(1, 1)`, `(2, 4)`, `(4, 5)` |

## State, Action, and Reward Design

The state representation is `(row, col, battery, rescued_mask)`. The `rescued_mask` is a bit mask, where bit `i = 1` means rescue target `i` has already been rescued. This captures drone position, battery level, and target rescue status.

The action space contains `UP`, `DOWN`, `LEFT`, `RIGHT`, and `HOVER`. Movement actions consume one battery unit. Blocked-cell and boundary movements keep the drone in the same position but still consume battery. Hovering on a charging station increases battery by `+2` up to maximum capacity, while hovering elsewhere consumes one battery unit.

Rewards follow the assignment structure: rescue target `+20`, danger zone `-10`, battery exhausted `-20`, regular movement `-1`, and charging benefit when the drone reaches a charging station for useful recharge. Charging stations always refill battery; the charging reward is guarded against repeated high-battery cycling so the policy focuses on rescue efficiency rather than reward farming.

## Value Iteration Evidence

| Metric | Value |
|---|---:|
| Approximate state count | 3168 |
| Reachable states enumerated | 1852 |
| Discount factor | 0.95 |
| Stopping threshold | 0.001 |
| Convergence iterations | 19 |
| Final delta/error | 0.000405 |
| Runtime seconds | 0.3608 |

The representative optimal trajectory rescues all three targets. It first moves toward the lower charging station, rescues the middle target, reaches the upper charging station when battery is low, rescues the upper-right target, recharges again, and then moves down to rescue the final lower target.

## State-Value Heatmap Interpretation

The heatmap fixes battery at full capacity and keeps all rescue targets unrescued. Higher state values appear near rescue targets because those cells lead quickly to the `+20` rescue reward. Lower values appear near danger zones due to the `-10` penalty and near blocked cells because movement choices are restricted. Charging stations are useful because they restore battery and allow longer rescue routes.

## Scalability Discussion

The approximate state count is `grid_cells * battery_levels * 2^(number_of_rescue_targets)`. For the current Team 108 setup, this is `36 * 11 * 8 = 3168`. If the grid becomes `10x10`, the battery capacity increases to `20`, and there are `6` rescue targets, the state count becomes approximately `100 * 21 * 64 = 134400` before adding weather state variables. This is the curse of dimensionality: every new rescue target doubles the rescue-status space. Classical DP is suitable for this small known model, but larger real-world drone systems require approximate methods such as Deep RL to generalize across states without explicitly storing every value.
