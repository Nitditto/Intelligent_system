# Assignment 1: Diabetes

# Report
1. Introduction
2. System definition
- System statement: 
	- Write a short paragraph describing the system
	- Draw system diagram showing: Input -> Represent -> Model -> Predict
3. Problem definition:
	- What real-world problem does the system address? 
	- What information does the system receive? 
	- How is that information represented internally? 
	- What does the model learn? 
	- What decision or prediction does it produce? 
	- Who or what uses the prediction?
4. Dataset
- Record the dataset source
- Answer questions:
	- What real-world phenomenon is represented?
	- What is one observation?
	- What are the features?
	- What is the target?
	- Is the target numerical or categorical? 
	- Is this regression or classification?
	- How many observations are available?
	- How many features are available?
	- Which features are numerical?
	- Which features are categorical?
5. Data representation
- Provide a table

| Feature | Type        | Representation | Meaning         |
| ------- | ----------- | -------------- | --------------- |
| Age     | Numerical   | Real value     | Age of subject  |
| Gender  | Categorical | Encoded value  | Category        |
| BMI     | Numerical   | Real value     | Body mass index |
Explain how categorical variables are represented numerically when required. (e.g. Gender: 0 - male, 1 - female)

6. Traditional ML Methods
- List what models to use (must have at least 4 models)
- For every selected model, explain:
	- What representation does it receive?
	- What relationship does it try to learn?
	- What parameters or structures are learned?
	- What criterion guides learning?
	- What assumptions does the model make?
	- What are its strengths?
	- What are its weaknesses?
7. Experimental Design
Students must conduct at least three controlled experiments.
- Experiment Model Comparison
	- Compare at least four models under the same evaluation protocol. 
	
| Model   | Accuracy | Precision | Recall | F1  |
| ------- | -------- | --------- | ------ | --- |
| Model 1 | -        | -         | -      | -   |
| Model 2 | -        | -         | -      | -   |
- Experiment Hyperparameter Investigation 
	- Change one meaningful hyperparameter. 
	- Examples:
		- k in k-NN;
		- tree depth;
		- number of trees;
		- SVM regularization;
		- regularization in Logistic Regression. 
	- The experimental question must be stated before running the experiment.
- Experiment Representation / Feature Investigation
	- This experiment directly responds to Slide 01. 
	- Compare, for example: $X_{all}$ vs. $X_{selected}$, or: unscaled features vs. standardized features. 
	- Students must explain whether the representation change affects the result and why.

8. Results
- For classification, report:
	- Accuracy;
	- Precision;
	- Recall;
	- F1-score;
	- Confusion Matrix. 
- For regression, report:
	- MAE;
	- MSE;
	- RMSE;
	- R2. 
- Students must explain which metrics are appropriate for their application.

9. Model Comparison
- Same as experimental model comparison of chapter 7
10. Representation Analysis
- Students must answer:
	- Why is your feature-vector representation appropriate?
	- What information does it preserve?
	- What information might it lose?
	- Could the same problem be represented as an image?
	- Could it be represented as a sequence?
	- Could it be represented as a graph?
	- Could it be represented using learned embeddings?
	- What would change if the representation changed?
11. Intelligent Application
- Capture images of the application with description of what it does
12. Limitations
- Limitations of the model / system
13. Reflection
- Students must answer: 
	- What information does your system receive?
	- What is the internal representation?
	- What does the model learn from examples?
	- What prediction or decision does it make?
	- Why can it handle an unseen input?
	- What part of the system can reasonably be called “intelligent”? 
	- What limitations prevent it from being a more capable intelligent system?
14. Conclusion


# Notebook structure
1. System and Problem Definition 
2. Intelligent System Diagram 
3. Dataset Source 
4. Dataset Description 
5. Data Representation 
6. Feature and Target Analysis 
7. Exploratory Data Analysis 
8. Train/Test Split 
9. Baseline 
10. Model 1 
11. Model 2 
12. Model 3 
13. Model 4 
14. Evaluation 
15. Experiment 1: Model Comparison 
16. Experiment 2: Hyperparameter Investigation 
17. Experiment 3: Representation / Feature Investigation 
18. Final Model 
19. Application 
20. System Demonstration 
21. Reflection 
22. Conclusion
## Requirements
- R1. Real Dataset: Use a real dataset from Kaggle or another approved public source. 
- R2. System Definition: Define the real-world problem and the intended intelligent system. 
- R3. Representation: Explicitly describe how raw information becomes model input. 
- R4. Feature Analysis: Explain the meaning and type of important features. 
- R5. Problem Formulation: Clearly define input, target, and learning task. 
- R6. Baseline: Implement and report a baseline. 
- R7. Multiple Models: Train at least four traditional ML models. 
- R8. Controlled Experiments: Perform at least three experiments. 
- R9. Evaluation: Use appropriate metrics and explain them. 
- R10. Scientific Analysis: Explain why the results were obtained. 
- R11. Application: Develop a small application using the selected model. 
- R12. System Demonstration: Demonstrate the complete input-to-prediction pipeline. 
- R13. Reflection: Explain the role and limitations of the representation. 
- R14. Reproducibility: The notebook must run from beginning to end with documented environment and dataset information.

