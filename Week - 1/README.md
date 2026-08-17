<div align="center">

# Week 1 – Pandas Basics & Data Cleaning

**EDP Internship · Foundation Phase**

A beginner-level data preprocessing assignment focused on learning **Pandas fundamentals**, understanding **features and labels**, and performing basic dataset exploration and cleaning.

</div>

---

## Week 1 Overview

| Item        | Details                              |
| :---------- | :----------------------------------- |
| Phase       | Foundation – Machine Learning Basics |
| Week        | 1 of 12                              |
| Topic       | Pandas Basics, Features & Labels     |
| Deliverable | Load and Clean a Sample Dataset      |
| Dataset     | Student Performance Dataset          |
| Language    | Python                               |
| Library     | Pandas                               |
| Environment | Jupyter Notebook                     |

### Objective

The objective of Week 1 was to understand the fundamentals of working with datasets using Pandas and learn how raw data can be explored, cleaned, and prepared for future Machine Learning tasks.

The workflow followed in this assignment was:

```text
Load
  ↓
Explore
  ↓
Identify Problems
  ↓
Clean
  ↓
Verify
  ↓
Save
```

---

## Project Structure

```text
Week-1-Pandas-Basics/
│
├── dataset/
│   └── student_data.csv
│
├── cleaned_dataset/
│   └── cleaned_student_data.csv
│
├── notebooks/
│   └── Data_Cleaning.ipynb
│
├── src/
│   └── data_cleaning.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

A small **Student Performance Dataset** was created for this assignment to demonstrate fundamental data loading, exploration, & cleaning techniques.

The original dataset contained **16 records and 7 columns**.

### Dataset Attributes

| Column             | Description                                 | Data Type |
| :----------------- | :------------------------------------------ | :-------: |
| `Student_ID`       | Unique identification number of the student |  Integer  |
| `Name`             | Name of the student                         |    Text   |
| `Gender`           | Gender of the student                       |    Text   |
| `Attendance`       | Attendance percentage                       |  Integer  |
| `Study_Hours`      | Average study hours                         |  Decimal  |
| `Assignment_Score` | Assignment score                            |  Decimal  |
| `Final_Score`      | Final examination score                     |  Integer  |

---

## Features & Labels

One of the fundamental Machine Learning concepts introduced during Week 1 was the distinction between **features** and **labels**.

### Features

Features are the input variables that provide information to a Machine Learning model.

For this dataset, the following columns can be considered features:

* `Student_ID`
* `Gender`
* `Attendance`
* `Study_Hours`
* `Assignment_Score`

### Label

The label is the target variable that a Machine Learning model may be trained to predict.

```text
Final_Score → Label / Target
```

For example, in a future Machine Learning task, the student's other attributes could be used as inputs to predict their `Final_Score`.

> **Note:** No Machine Learning model was trained during Week 1. Features and labels were introduced as foundational concepts for the upcoming weeks.

---

## Pandas Concepts Learned

The following Pandas operations were practiced during this assignment:

| Operation              | Purpose                                       |
| :--------------------- | :-------------------------------------------- |
| `pd.read_csv()`        | Load a CSV dataset                            |
| `df.head()`            | Display the first few records                 |
| `df.tail()`            | Display the last few records                  |
| `df.shape`             | Determine the number of rows and columns      |
| `df.columns`           | View column names                             |
| `df.dtypes`            | Check data types                              |
| `df.info()`            | Inspect dataset structure and non-null values |
| `df.describe()`        | Generate statistical summaries                |
| `df.isnull()`          | Identify missing values                       |
| `df.isnull().sum()`    | Count missing values                          |
| `df.fillna()`          | Replace missing values                        |
| `df.mean()`            | Calculate the mean of numerical data          |
| `df.duplicated()`      | Identify duplicate records                    |
| `df.drop_duplicates()` | Remove duplicate records                      |
| `df.to_csv()`          | Save the processed dataset                    |

---

# Dataset Exploration

## Loading the Dataset

The dataset was loaded into a Pandas DataFrame using:

```python
import pandas as pd

df = pd.read_csv("../dataset/student_data.csv")
```

A **DataFrame** is Pandas' primary two-dimensional data structure and represents data in rows and columns.

---

## Exploring the Dataset

The following operations were used to understand the structure and characteristics of the dataset.

### View Dataset

```python
df
```

### View First Five Records

```python
df.head()
```

### View Last Five Records

```python
df.tail()
```

### Check Dataset Dimensions

```python
df.shape
```

Initial dataset shape:

```text
(16, 7)
```

This represents:

| Property | Value |
| :------- | ----: |
| Rows     |    16 |
| Columns  |     7 |

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

These operations provided an initial understanding of the dataset structure, data types, missing values, and numerical characteristics.

---

# Data Cleaning

The original dataset contained intentional data quality issues so that common preprocessing techniques could be practiced.

## Issues Identified

| Issue            | Column / Record    | Count |
| :--------------- | :----------------- | ----: |
| Missing value    | `Study_Hours`      |     1 |
| Missing value    | `Assignment_Score` |     1 |
| Duplicate record | Complete row       |     1 |

---

## Handling Missing Values

Missing values were first identified using:

```python
df.isnull()
```

The number of missing values in each column was then calculated using:

```python
df.isnull().sum()
```

The missing numerical values were handled using **mean imputation**, where each missing value was replaced with the mean of its respective column.

### Study Hours

```python
df["Study_Hours"] = df["Study_Hours"].fillna(
    df["Study_Hours"].mean()
)
```

### Assignment Score

```python
df["Assignment_Score"] = df["Assignment_Score"].fillna(
    df["Assignment_Score"].mean()
)
```

After this operation, the dataset contained no missing values in these columns.

---

## Handling Duplicate Records

Duplicate records were identified using:

```python
df.duplicated()
```

The number of duplicate records was checked using:

```python
df.duplicated().sum()
```

One duplicate record was identified.

The duplicate was removed using:

```python
df = df.drop_duplicates()
```

The result was verified using:

```python
df.duplicated().sum()
```

The result after cleaning was:

```text
0
```

Therefore, no duplicate records remained.

---

# Final Dataset

The dataset changed from:

| Stage           | Rows | Columns |
| :-------------- | ---: | ------: |
| Before Cleaning |   16 |       7 |
| After Cleaning  |   15 |       7 |

### Final Data Quality

| Check             | Result |
| :---------------- | :----: |
| Missing values    |    0   |
| Duplicate records |    0   |
| Valid records     |   15   |
| Columns           |    7   |

The cleaned dataset was saved as:

```text
cleaned_student_data.csv
```

inside the `cleaned_dataset/` directory.

---

## Saving the Cleaned Dataset

The cleaned DataFrame was exported using:

```python
df.to_csv(
    "../cleaned_dataset/cleaned_student_data.csv",
    index=False
)
```

The `index=False` parameter prevents Pandas from adding the DataFrame index as an unnecessary column in the exported CSV file.

---

# Implementation

The complete assignment was implemented in two formats.

| Implementation   | File                            | Purpose                              |
| :--------------- | :------------------------------ | :----------------------------------- |
| Jupyter Notebook | `notebooks/Data_Cleaning.ipynb` | Interactive exploration and learning |
| Python Script    | `src/data_cleaning.py`          | Reusable data cleaning workflow      |

### Data Processing Workflow

```text
Student Dataset
      ↓
Load using Pandas
      ↓
Explore Dataset
      ↓
Identify Missing Values
      ↓
Fill Missing Values
      ↓
Identify Duplicate Records
      ↓
Remove Duplicates
      ↓
Verify Data Quality
      ↓
Export Cleaned Dataset
```

---

## Jupyter Notebook

The complete interactive implementation is available in:

```text
notebooks/Data_Cleaning.ipynb
```

The notebook contains the step-by-step process of loading, exploring, cleaning, and verifying the dataset.

---

## Python Script

The reusable implementation is available in:

```text
src/data_cleaning.py
```

The script performs the complete data processing workflow and generates the cleaned dataset.

---

## Technologies & Tools

| Technology / Tool | Purpose                                     |
| :---------------- | :------------------------------------------ |
| Python            | Primary programming language                |
| Pandas            | Data manipulation and cleaning              |
| Jupyter Notebook  | Interactive development and experimentation |
| Git               | Version control                             |
| GitHub            | Repository management                       |

---

## Skills Learned

By completing Week 1, the following skills were developed:

* Understanding Pandas fundamentals
* Working with Pandas DataFrames
* Loading CSV datasets
* Exploring dataset structure
* Understanding rows and columns
* Identifying numerical and categorical data
* Understanding features and labels
* Detecting missing values
* Handling missing values using mean imputation
* Detecting duplicate records
* Removing duplicate records
* Verifying cleaned data
* Exporting processed datasets
* Creating reusable Python data-processing scripts
* Organizing a data science project
* Using Git and GitHub for version control

---

## Key Learning

The major takeaway from Week 1 was understanding that **data preprocessing is an essential step before applying Machine Learning algorithms**.

Real-world datasets may contain missing values, duplicate records, and other data quality issues. Pandas provides simple and powerful tools for identifying and handling these problems.

The concepts learned in this assignment establish the foundation for the Machine Learning tasks that follow in the upcoming weeks.

---

## Week 1 Outcome

The **Load and Clean a Sample Dataset** assignment was successfully completed.

The raw Student Performance Dataset was:

**Loaded → Explored → Cleaned → Verified → Exported**

The final dataset contains **15 records and 7 columns**, with missing values and duplicate records successfully handled.

This assignment establishes the data handling foundation required for the next stage of the internship: **training the first regression model in Week 2**.

---


## Author

**T. Navaneeth Reddy**

B.Tech – Information Technology
Institute of Aeronautical Engineering

**GitHub:** [Navaneeth4141](https://github.com/Navaneeth4141)

---

<div align="center">

### EDP Internship · Week 1

**Learn • Implement • Analyze • Document**

</div>
