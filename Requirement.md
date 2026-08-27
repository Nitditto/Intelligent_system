# Report template
- **Should be 10 pages**
## Cover page
```
INTELLIGENT SYSTEM DEVELOPMENT
ASSIGNMENT 02
From Data Representation to Deployable Intelligent Systems
Student name: ................................................
Student ID: ................................................
Class: ................................................
Team: ................................................
Lecturer: Dinh Que Tran, Ph.D., Assoc. Prof.
Semester: I.2026
```
## Executive summary

|Application|Prediction/Task|Main Representation|
|---|---|---|
|Diabetes|Diabetes classification|Feature vector/matrix|
|House Price|Price regression|Feature vector/matrix|
|Customer Behavior|Behavior / interest prediction|Feature vector / matrix|

- Diabetes
    - Dataset: 
    - Problem: 
    - Representation:
    - Selected model:
    - Main eval result:
    - Deployment method:

- House price
    - Dataset:
    - ...
## Data Representation Overview

|Application|Raw Data|ML Representation|
|---|---|---|
|Diabetes|CSV|feature matrix|
|House Price|CSV|feature matrix|
|Customer Behaviour|CSV/transactions|feature matrix/sequences|

- Diabetes:
    - What does one row represent?
    - What does one column represent?
    - Which columns are input features?
    - Which column is the target?
    - Which features are numerical?
    - Which features are categorical?
    - How are categorical values encoded?
    - What is the final feature dimension?
    - What is the shape of the model input?

*(same with the other 2 applications)*
- Model input: $X \in \R^{N \times d}$, where:
    - $N$: number of rows
    - $d$: number of columns
- Feature dimension: $y \in \R^{N}$
## 