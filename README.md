# DRL Assignment 1 - Team 108

This project contains the implementation work for both parts of Deep Reinforcement Learning Lab Assignment 1.

## Team Details

- Group number: 108
- Team members: ARTHIKA G ., SRINEVEDA R S ., ASWATHY H ., AMIYA PALAI ., KUTAREKAR NISHCHAL AJAY .

## Project Structure

```text
DRL_ASSIGNMENT_1/
  ASSIGNMENT_1_DESCRIPTION.txt
  README.md
  requirements.txt
  solutions/
    Team_108_MAB.py
    Team_108_DP.py
  outputs/
    Team_108_MAB_iteration_log.csv
    Team_108_MAB_summary.csv
    Team_108_DP_policy_trajectory.csv
    Team_108_DP_value_iteration_summary.csv
    figures/
      Team_108_MAB_cumulative_reward.png
      Team_108_DP_initial_grid.png
      Team_108_DP_policy_grid.png
      Team_108_DP_value_heatmap.png
      Team_108_DP_trajectory.png
  reports/
    Team_108_MAB_Report_Notes.md
    Team_108_DP_Report_Notes.md
```

## How To Run

On this machine, the Windows Python launcher works as `py`. If `python` is available in the virtual lab, either command is fine.

```powershell
py -m pip install -r requirements.txt
py .\solutions\Team_108_MAB.py
py .\solutions\Team_108_DP.py
```

or:

```powershell
python -m pip install -r requirements.txt
python .\solutions\Team_108_MAB.py
python .\solutions\Team_108_DP.py
```

Both scripts print the execution timestamp, virtual machine ID, machine name, group number, and team members at the top of the output.

## PDF Submission Workflow

1. Run `Team_108_MAB.py` in the virtual lab and capture the output with timestamp visible.
2. Insert the MAB code, output tables, and `outputs/figures/Team_108_MAB_cumulative_reward.png` into the Part 1 PDF.
3. Run `Team_108_DP.py` in the virtual lab and capture the output with timestamp visible.
4. Insert the DP code, convergence output, trajectory output, and all DP figures into the Part 2 PDF.
5. Use the files in `reports/` as concise written explanation text for the final PDFs.

## Group 108 Derived Parameters

For MAB:

- Number of medicines: `(108 mod 3) + 5 = 5`
- Hidden success probabilities: `[0.40, 0.47, 0.54, 0.61, 0.68]`
- True best medicine: medicine `4`

For DP:

- Last digit is `8`, so grid size is `6x6`
- Maximum battery is `10` because `8` is even
- Wind probability is `30%`
- Maximum step limit is `75`
- Rescue targets: `3`
- Charging stations: `2`
- Danger zones: `4`
- Blocked cells: `3`
