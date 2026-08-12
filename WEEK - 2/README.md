<div align="center">

# Week 2 – Linear Regression & Model Evaluation

**EDP Internship · Foundation Phase**

A beginner-level Machine Learning assignment focused on learning **Linear Regression**, understanding **Train/Test Split**, training the first regression model, evaluating model performance using regression metrics, and visualizing the predictions.

</div>

---

## Week 2 Overview

| Item        | Details                                                  |
| :---------- | :------------------------------------------------------- |
| Phase       | Foundation – Machine Learning Basics                     |
| Week        | 2 of 12                                                  |
| Topic       | Linear Regression, Train/Test Split & Evaluation Metrics |
| Deliverable | Train and Evaluate the First Regression Model            |
| Dataset     | Salary Prediction Dataset                                |
| Language    | Python                                                   |
| Libraries   | Pandas, NumPy, Matplotlib, Scikit-learn                  |
| Environment | Jupyter Notebook                                         |

### Objective

The objective of Week 2 was to understand the fundamentals of **Linear Regression**, learn how to divide a dataset into training and testing sets, train the first Machine Learning regression model, generate predictions, evaluate the model using standard regression metrics, and visualize the learned regression relationship.

The workflow followed in this assignment was:

```text
Load
  ↓
Explore
  ↓
Select Features & Target
  ↓
Train/Test Split
  ↓
Train Model
  ↓
Predict
  ↓
Evaluate
  ↓
Visualize
```

---

## Project Structure

```text
Week-2-Linear-Regression/
│
├── dataset/
│   └── salary_data.csv
│
├── notebooks/
│   └── Linear_Regression.ipynb
│
├── outputs/
│   ├── regression_plot.png
│   └── evaluation_results.txt
│
├── src/
│   └── linear_regression.py
│
├── requirements.txt
└── README.md
```

---

## Dataset

A small **Salary Prediction Dataset** was created for this assignment to demonstrate the fundamentals of regression and model training.

The dataset contains **30 records and 2 columns**.

### Dataset Attributes

| Column                | Description                                | Data Type |
| :-------------------- | :----------------------------------------- | :-------: |
| `Years_of_Experience` | Number of years of professional experience |  Decimal  |
| `Salary`              | Corresponding annual salary                |  Integer  |

---

## Features & Target

One of the important concepts practiced during Week 2 was separating the dataset into **features** and a **target variable** before training the Machine Learning model.

### Feature

The feature is the input variable provided to the model.

```text
Years_of_Experience → Feature / Input
```

### Target

The target is the output variable that the model is trained to predict.

```text
Salary → Target / Label
```

The feature and target were separated using:

```python
X = df[["Years_of_Experience"]]
y = df["Salary"]
```

---

## Machine Learning Concepts Learned

The following concepts were practiced during this assignment:

| Concept           | Purpose                                                     |
| :---------------- | :---------------------------------------------------------- |
| Linear Regression | Model the linear relationship between experience and salary |
| Features          | Represent the input variables used by the model             |
| Target            | Represent the value the model needs to predict              |
| Train/Test Split  | Divide data for training and evaluation                     |
| Model Training    | Allow the model to learn from training data                 |
| Prediction        | Generate salary values for unseen test data                 |
| MAE               | Measure average absolute prediction error                   |
| MSE               | Measure average squared prediction error                    |
| RMSE              | Measure prediction error in the original target unit        |
| R² Score          | Measure how well the model explains target variation        |
| Visualization     | Display actual values and the regression line               |

---

# Dataset Exploration

## Loading the Dataset

The dataset was loaded into a Pandas DataFrame using:

```python
import pandas as pd

df = pd.read_csv("../dataset/salary_data.csv")
```

The dataset was then inspected to understand its structure and contents.

---

## Exploring the Dataset

The following operations were used during the initial exploration:

### View Dataset

```python
df
```

### View First Five Records

```python
df.head()
```

### Check Dataset Dimensions

```python
df.shape
```

Initial dataset shape:

```text
(30, 2)
```

This represents:

| Property | Value |
| :------- | ----: |
| Rows     |    30 |
| Columns  |     2 |

### View Column Names

```python
df.columns
```

### Check Data Types

```python
df.dtypes
```

### View Dataset Information

```python
df.info()
```

### Generate Statistical Summary

```python
df.describe()
```

### Check Missing Values

```python
df.isnull().sum()
```

### Check Duplicate Records

```python
df.duplicated().sum()
```

These operations provided an understanding of the dataset structure and confirmed that the dataset was suitable for training the regression model.

---

# Relationship Between Features and Target

Before training the model, the relationship between **Years of Experience** and **Salary** was visualized using a scatter plot.

```python
plt.scatter(
    df["Years_of_Experience"],
    df["Salary"]
)
```

The visualization showed a clear positive relationship between the two variables.

```text
Years of Experience ↑
        ↓
Salary generally ↑
```

This made Linear Regression suitable for demonstrating the prediction task.

---

# Train/Test Split

The dataset was divided into training and testing sets using Scikit-learn's `train_test_split()` function.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The dataset was divided as follows:

| Dataset       | Percentage | Records |
| :------------ | :--------: | ------: |
| Training Data |     80%    |      24 |
| Testing Data  |     20%    |       6 |
| Total         |    100%    |      30 |

### Why Train/Test Split?

The training dataset is used to teach the model the relationship between the feature and target.

The testing dataset is kept separate and is used to evaluate the model on data that was not used during training.

A `random_state` of `42` was used to make the splitting process reproducible.

---

# Linear Regression Model

A Linear Regression model from Scikit-learn was created and trained using the training dataset.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

The `fit()` method allows the model to learn the relationship between **Years of Experience** and **Salary** from the training data.

---

## Linear Regression Equation

After training, the model learned the following equation:

```text
Salary = 9600.41 × Years_of_Experience + 27598.02
```

### Coefficient

```text
9600.41
```

The coefficient represents the estimated change in salary for every additional year of experience.

According to this model, salary increases by approximately **₹9,600.41** for each additional year of experience.

### Intercept

```text
27598.02
```

The intercept represents the model's estimated salary when the years of experience is zero.

---

# Making Predictions

The trained model was used to predict salary values for the testing dataset.

```python
y_pred = model.predict(X_test)
```

The predicted values were then compared with the actual salary values from the testing dataset.

---

## Actual vs Predicted Values

A comparison table was created using Pandas:

```python
comparison = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": y_pred
})
```

This allowed the predictions produced by the model to be directly compared with the actual salary values.

The predicted values were very close to the actual values in the testing dataset.

---

# Model Evaluation

The trained regression model was evaluated using four standard regression metrics:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## Evaluation Metrics

### Mean Absolute Error – MAE

MAE represents the average absolute difference between the actual and predicted values.

```text
MAE = 419.46
```

The model's predictions differ from the actual salary values by approximately **₹419 on average**.

---

### Mean Squared Error – MSE

MSE calculates the average squared difference between actual and predicted values.

```text
MSE = 223215.72
```

Squaring the errors gives larger errors more influence on the final metric.

---

### Root Mean Squared Error – RMSE

RMSE is the square root of MSE and expresses the prediction error in the same unit as the target variable.

```text
RMSE = 472.46
```

Therefore, the typical prediction error represented by RMSE is approximately **₹472**.

---

### R² Score

R² measures how well the model explains the variation in the target variable.

```text
R² = 0.9993
```

This means that the model explains approximately **99.93% of the variation in salary** on the test dataset.

> **Note:** R² should not be interpreted as "99.93% accuracy." It represents the proportion of variance explained by the regression model.

---

## Final Model Evaluation

| Metric |    Result |
| :----- | --------: |
| MAE    |    419.46 |
| MSE    | 223215.72 |
| RMSE   |    472.46 |
| R²     |    0.9993 |

The model achieved a very high R² score and low prediction errors on the testing dataset, indicating a strong linear relationship between years of experience and salary in the synthetic dataset.

---

# Regression Visualization

The actual testing values and the fitted Linear Regression line were visualized using Matplotlib.

```python
plt.scatter(
    X_test,
    y_test,
    label="Actual Salary"
)

plt.plot(
    X_test_sorted,
    y_pred_sorted,
    label="Regression Line"
)
```

The final visualization is saved as:

```text
outputs/regression_plot.png
```

The test data points appear very close to the regression line, which visually supports the strong performance obtained from the evaluation metrics.

---

# Output Files

The project generates the following output files:

| File                     | Purpose                                                            |
| :----------------------- | :----------------------------------------------------------------- |
| `regression_plot.png`    | Regression visualization showing actual values and the fitted line |
| `evaluation_results.txt` | Stores the model equation and evaluation metrics                   |

The Python implementation automatically generates and updates these files whenever the program is executed.

---

# Implementation

The complete assignment was implemented in two formats.

| Implementation   | File                                | Purpose                                  |
| :--------------- | :---------------------------------- | :--------------------------------------- |
| Jupyter Notebook | `notebooks/Linear_Regression.ipynb` | Interactive learning and experimentation |
| Python Script    | `src/linear_regression.py`          | Reusable regression workflow             |

### Machine Learning Workflow

```text
Salary Dataset
      ↓
Load using Pandas
      ↓
Explore Dataset
      ↓
Select Feature & Target
      ↓
Train/Test Split
      ↓
Create Linear Regression Model
      ↓
Train Model
      ↓
Generate Predictions
      ↓
Evaluate Model
      ↓
Visualize Regression Line
      ↓
Save Results
```

---

## Jupyter Notebook

The complete interactive implementation is available in:

```text
notebooks/Linear_Regression.ipynb
```

The notebook contains the step-by-step implementation of dataset exploration, feature and target selection, train/test splitting, model training, prediction, evaluation, and visualization.

---

## Python Script

The reusable implementation is available in:

```text
src/linear_regression.py
```

The script performs the complete regression workflow and automatically generates the evaluation results and regression visualization.

---

# Technologies & Tools

| Technology / Tool | Purpose                                     |
| :---------------- | :------------------------------------------ |
| Python            | Primary programming language                |
| Pandas            | Data loading and manipulation               |
| NumPy             | Numerical calculations                      |
| Matplotlib        | Data visualization                          |
| Scikit-learn      | Machine Learning and model evaluation       |
| Jupyter Notebook  | Interactive development and experimentation |
| Git               | Version control                             |
| GitHub            | Repository management                       |

---

# Running the Project

## Install Dependencies

Install the required libraries using:

```bash
pip install -r requirements.txt
```

## Run the Python Script

From the Week 2 directory:

```bash
python src/linear_regression.py
```

The program will:

1. Load the salary dataset
2. Separate features and target
3. Split the dataset into training and testing sets
4. Train the Linear Regression model
5. Generate predictions
6. Calculate MAE, MSE, RMSE, and R²
7. Save the evaluation results
8. Generate the regression visualization

## Run the Jupyter Notebook

Open:

```text
notebooks/Linear_Regression.ipynb
```

and execute the cells sequentially.

---

# Skills Learned

By completing Week 2, the following skills were developed:

* Understanding the fundamentals of Linear Regression
* Identifying features and target variables
* Splitting datasets into training and testing sets
* Training a Machine Learning regression model
* Generating predictions using a trained model
* Understanding regression coefficients and intercepts
* Evaluating regression models using MAE, MSE, RMSE, and R²
* Interpreting model evaluation results
* Visualizing regression results using Matplotlib
* Using Scikit-learn for Machine Learning
* Creating reusable Python Machine Learning scripts
* Organizing a Machine Learning project
* Generating reproducible model outputs
* Using Git and GitHub for version control

---

# Key Learning

The major takeaway from Week 2 was understanding the basic workflow of building a Machine Learning regression model.

The assignment demonstrated that a Machine Learning workflow involves more than simply training a model. The data must first be explored, the features and target must be identified, the dataset must be divided into training and testing sets, and the trained model must be evaluated using appropriate metrics.

The project also provided practical understanding of how **MAE, MSE, RMSE, and R²** provide different perspectives on model performance.

---

# Week 2 Outcome

The **Train First Regression Model** assignment was successfully completed.

The Salary Prediction Dataset was:

**Loaded → Explored → Prepared → Split → Trained → Predicted → Evaluated → Visualized**

A Linear Regression model was successfully trained using **80% of the dataset** and evaluated using the remaining **20%**.

The final model achieved:

```text
MAE  : 419.46
MSE  : 223215.72
RMSE : 472.46
R²   : 0.9993
```

The Week 2 assignment established the foundation for building and evaluating Machine Learning models and prepares for more advanced Machine Learning concepts in the upcoming weeks.

---

## Author

**T. Navaneeth Reddy**

B.Tech – Information Technology

Institute of Aeronautical Engineering

**GitHub:** [Navaneeth4141](https://github.com/Navaneeth4141)

---

<div align="center">

### EDP Internship · Week 2

**Learn • Implement • Analyze • Document**

</div>
