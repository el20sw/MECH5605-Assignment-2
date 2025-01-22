# Git Repository for MECH5605M: Biomechatronics and Medical Robotics - Assignment 2

## How to use this repository

This repository contains the code for the second assignment of the course MECH5605M: Biomechatronics and Medical Robotics. The code is written in MATLAB.

1. Running `PreProcessing.m` takes the raw IMU data and converts it into a form suitable for extracting features
2. Running `FeatureExtraction.m` extracts features from the data using the sliding window technique for use in classification
3. `ANN.m`, `SVM.m`, `CNN.m` and `RF.m` contain the implementations of the Artifical Neural Network, Support Vector Machine, Convolutional Neural Network, and Random Forest Ensemble respectively
4. `FeatureSelection.m` runs various feature selection methods including MRMR and brute force ANN methods
5. `SegmentSelection.m` finds the most significant segment through the use of multiple ANNs trained with the same hyperparameters to find the most significant segment

### Cloning the repository

To clone the repository, use the following command in your terminal:

```bash
git clone https://github.com/el20sw/MECH5605-Assignment-2.git
```

## Structure
The repository includes:
- Code files: MATLAB scripts for classification and machine learning tasks
- IMU Data: Revised IMU datasets for testing (modified file and directory names for ease of use)

### Collaboration

Please follow Git best practices:
- **Branching**: Create seperate branches for adding new features and tasks
- **Commits**: Write clear, descriptive commit messages summarising any changes
- **Pull Requests**: We can try to use Pull Requests as a way of collaboration and ensuring changes are quality

## Task:

The students will work in their teams to filter the data and extract time domain features
from the collected data (IMUs). These time domain features are:

1. Maximum
2. Minimum
3. Mean
4. Standard deviation
5. Root means square (RMS)
6. Maximum gradient
7. Number of zero crossings

The overlapping, sliding window time-series segmentation strategy will be used to divide the
gait signals into a sequence of overlapped discrete segments. The time domain features will
be calculated for each time window. Each team will calculate these features from the raw data
for segmentation windows of tw with increments of $\Delta t$ in milliseconds as shown in Table 1.

Features will be extracted from each analysis window. After analysing the data and selection
of the important features, the following classification techniques can be used for detecting the
user intent for five different activities (i.e., Level ground walking at normal pace, Ramp
ascending, Ramp descending, Sit to stand and Stand to sit as mentioned before):

1. Neural Network Classification
2. Support Vector Machines (SVM)
3. Convolutional Neural Networks (Deep Learning)
4. Another classification method of your choice

It is required to find:

1. Classification error for the used features;
2. The most significant fifteen features;
3. The most significant segment which contributes to the activities recognition;
4. Using only the features from the significant segment (identified in 3), find the classification error.
5. Compare the results of the different classification techniques using confusion matrices.

### Table 1. Segmentation window size and increments for each group
| Group no. | Window size ($t_w$) | Increment ($\Delta t$) |
|-----------|------------------|----------------|
| 1         | 300              | 50             |
| 2         | 400              | 100            |
| 3         | 350              | 100            |
| 4         | 300              | 100            |
| 5         | 500              | 30             |
| 6         | 350              | 50             |
| 7         | 400              | 50             |
| 8         | 450              | 50             |
| 9         | 450              | 70             |
| 10        | 350              | 70             |
| 11        | 300              | 70             |
| ***12***  | ***350***        | ***30***       |
| 13        | 400              | 70             |
| 14        | 450              | 100            |
| 15        | 300              | 30             |
| 16        | 350              | 80             |
| 17        | 500              | 50             |
| 18        | 500              | 70             |
| 19        | 500              | 80             |
| 20        | 400              | 30             |
| 21        | 400              | 80             |
