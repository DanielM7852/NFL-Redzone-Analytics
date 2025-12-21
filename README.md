# NFL Red Zone Play Sheet
![Catch Rate vs Distance](figures/catch_rate_distance.png) for images
---

## Overview


This project aims to identify optimal red zone play
strategies by analyzing broad offensive schemes and player kinematic tracking data.
In the end, we built a dashboard that allows coaches and players to make split-second decisions by identifying optimal play patterns 5–20 yards from the end zone, including recommended alignments, routes, and acceleration based on predicted defensive coverage. Coaches can also interactively adjust defender angle and distance relative to a targeted receiver to see how alignment affects catch probability.

---

## Objectives

- Discover and define metrics that lead to both a higher proportion of catches and touchdowns
- Create an interface that is intuitive and usable in real-world applications. 

---

## Data Sets

### Train
- 18 weeks of 0.1s frame intervals tracking every player for each play in every game.
- 18 input files which included all passing plays (before the ball was thrown) with stats on player postioning as well as player profiles and ball landing positon.
- 18 output files which included all passing plays (after the ball was released by QB) with stats on player positoning at frame intervals.

### Supplementary
- Additonal information on plays including catch result, formation (defense and offense), time elapsed, and play commentary.
  
---

## Methodolgy
- Joined input, output, and supplementary datasets and used computational analysis to define new features, including ball catch position, closest defender distance, and defender position relative to the targeted receiver.
- Calculated receiver acceleration using frame-by-frame displacement divided twice by the 0.1-second frame intervals, identified touchdown plays through string matching the keyword “TOUCHDOWN,” and binned the analysis into 5-yard recommendation intervals spanning 5–20 yards from the end zone.
- Explored the data to identify initial trends and develop testable hypotheses.
- Tested the significance of closer defender proximity in the red zone and found the effect to be statistically significant at the 0.01 level.
- Tested defender positioning and found that defenders aligned in front of the receiver were 10% more likely—statistically significant—to allow a completed catch than defenders positioned behind the receiver within the 0.5–2.5 yard proximity range.
- We used a Bayesian approach by defining a prior based on the global average touchdown rate, updating it with observed outcomes to form a posterior distribution, and estimating conservative touchdown probabilities using Monte Carlo simulation.
 

---

## Key Findings

- The distribution of catch frequency by defender proximity is highly right-skewed, meaning most catches occur close to a defender.
- This effect is even more pronounced in red zone plays, resulting in an even more extreme skew.


- At low seperation distances (0.5-2.5) yards, recivers classified as in "front of the defender" are expected to catch the ball 10% more often than under normal conditions.
- 


---

## Going Forward

Future enhancements could integrate quarterback performance metrics to account for passer accuracy, develop play sequencing recommendations to exploit defensive adjustments across drives, and expand beyond touchdown optimization to include success probabilities at any part of the field. The framework established here provides a robust foundation for ongoing NFL analytics innovation in a time where the power of data continues to increase.

---

## Project Links
- [pdf to Writeup](#)
- [Kaggle Notebook](https://www.kaggle.com/code/karstenlowe0/nfl-analytics-dashboard-final-version/edit)

---

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib
- Jupyter Notebook






