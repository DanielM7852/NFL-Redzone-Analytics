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


- Join the input, output, and supplementary data and use computational analysis to define new columns, including ball catch postion, closest defender distance, and defender position relative to the targeted reciever.
The  from every week were merged using game and play identifiers as join keys. To analyze catch behavior, we created catch-position columns by identifying the targeted receiver’s location within a 1-yard radius of the ball’s landing point, selecting the moment when the receiver was closest to the ball. We also calculated a defensive-distance metric, defined as the minimum Euclidean distance from the nearest defender to the receiver at the moment of the catch. For analysis purposes, defender distances were binned into intervals, allowing us to track catch rates across discrete distance ranges. Throws with a catch-time interval of less than 0.2 seconds were excluded, as such data is considered unreliable.
Red Zone plays were stratified into three intervals based on yardline distance from the end zone: 5–10 yards, 10–15 yards, and 15–20 yards. This grouping balances context and sample size: plays within the 5 are often run plays. A custom function mapped each play to its corresponding interval, yielding a filtered dataset of 1,982 unique Red Zone plays for analysis. 	For each unique play, we grouped the receiver formation and alignment, route type, quarterback snap position, and defensive coverage into distinct categories to perform statistical analysis. We also created flexibility with each receiver position by binning receiver alignment within the standard deviation across successful plays using the same formation and route combination.
Touchdown identification was performed using text pattern matching on play descriptions, searching for the keyword "TOUCHDOWN". Across 2,692 red zone plays analyzed, 318 resulted in touchdowns, yielding an overall baseline success rate of 11.8%. This baseline rate served as the prior probability to reference in later Bayesian simulation models.
	To calculate receiver acceleration patterns we computed frame-by-frame displacement on consecutive x-y positions, then derived instantaneous speed by dividing displacement by the 0.1 second frame interval. Acceleration was calculated by computing the rate of change of speed. To create an acceleration effort percentage, we divided by a realistic red zone top acceleration of 2.5 yards / sec². 
	Finally, we wanted to classify the defender position relative to the targeted receiver and the ball path at the catch. Using the ball path as a reference axis, we classified defender position by transforming the coordinate system so that the receiver was centered at the origin. Defender locations were expressed in this receiver-centric frame and mapped onto a unit-circle–like representation. We then computed the defender’s angular position relative to the ball path using the arctan2(y, x) function in Python, where x and y denote the defender’s coordinates relative to the receiver. This angle was used to categorize the defender as being ‘in front’, ‘behind’, or to the ‘side’ of the receiver. These categories were added as a new column, ‘d_pos’, to the dataframe. An abstraction of the field is shown below with an example of ‘d_pos’ categorization below.


In this example the defender would be labeled as ‘in front’ in the ‘d_pos’ column because the defender falls within the bounds of 115 to 245 degrees relative to the receiver and ball.
 

---

## Key Findings

- The distribution of catch frequency by defender proximity is highly right-skewed, meaning most catches occur close to a defender.
- This effect is even more pronounced in red zone plays, resulting in an even more extreme skew.


- At low seperation distances (0.5-2.5) yards, recivers classified as in "front of the defender" are expected to catch the ball 10% more often than under normal conditions.
- 
This analysis successfully transformed raw NFL tracking data into actionable coaching intelligence for red zone play calling. By combining spatial catch probability modeling with Bayesian statistical techniques and interactive visualization, the system provided coaches with reliable, data-driven play recommendations tailored to specific game situations.

---

## Going forward

Future enhancements could integrate quarterback performance metrics to account for passer accuracy, develop play sequencing recommendations to exploit defensive adjustments across drives, and expand beyond touchdown optimization to include success probabilities at any part of the field. The framework established here provides a robust foundation for ongoing NFL analytics innovation in a time where the power of data continues to increase.

---

## Project Links
- [pdf to Writeup](#)
- [Kaggle Notebook](#)

---

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib
- Jupyter Notebook






