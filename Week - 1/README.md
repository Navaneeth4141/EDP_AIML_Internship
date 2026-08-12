\# Week 1 – Pandas Basics \& Data Cleaning



A beginner-level data preprocessing project completed as part of the \*\*EDP Internship\*\*. This week's task focused on learning the fundamentals of \*\*Python Pandas\*\*, understanding \*\*features and labels\*\*, and performing basic data loading, exploration, and cleaning operations on a sample student performance dataset.



\---



\## 📌 Week 1 Objective



The objective of Week 1 was to build a basic understanding of data handling with Pandas and learn how raw datasets can be prepared before they are used for Machine Learning.



The project covers the complete basic workflow:



\*\*Load → Explore → Identify Problems → Clean → Verify → Save\*\*



\---



\## 📂 Project Structure



```text

Week-1-Pandas-Basics/

│

├── dataset/

│   └── student\_data.csv

│

├── cleaned\_dataset/

│   └── cleaned\_student\_data.csv

│

├── notebooks/

│   └── Data\_Cleaning.ipynb

│

├── src/

│   └── data\_cleaning.py

│

├── requirements.txt

├── .gitignore

└── README.md

```



\---



\## 📊 Dataset



For this project, a small \*\*Student Performance Dataset\*\* was created to demonstrate fundamental data cleaning techniques.



The dataset contains \*\*16 records and 7 columns\*\* before cleaning.



\### Dataset Attributes



| Column             | Description                                 | Type    |

| ------------------ | ------------------------------------------- | ------- |

| `Student\_ID`       | Unique identification number of the student | Integer |

| `Name`             | Name of the student                         | Text    |

| `Gender`           | Gender of the student                       | Text    |

| `Attendance`       | Attendance percentage                       | Integer |

| `Study\_Hours`      | Average study hours                         | Decimal |

| `Assignment\_Score` | Assignment score                            | Decimal |

| `Final\_Score`      | Final examination score                     | Integer |



\---



\## 🎯 Features and Label



An important concept introduced during Week 1 was the distinction between \*\*features\*\* and \*\*labels\*\*.



\### Features



Features are the input variables that provide information to a Machine Learning model.



For this dataset, examples of features are:



\* Student ID

\* Gender

\* Attendance

\* Study Hours

\* Assignment Score



\### Label



The label is the target variable that a Machine Learning model may be trained to predict.



In this dataset:



\*\*`Final\_Score` → Label / Target\*\*



For example, in a future Machine Learning task, the other student attributes could be used as inputs to predict the student's final score.



> Note: No Machine Learning model was trained in Week 1. Features and labels were introduced as a foundational Machine Learning concept.



\---



\## 🐼 Pandas Concepts Learned



The following Pandas operations were practiced during this week:



| Pandas Operation       | Purpose                                    |

| ---------------------- | ------------------------------------------ |

| `pd.read\_csv()`        | Load a CSV dataset                         |

| `df.head()`            | Display the first few records              |

| `df.tail()`            | Display the last few records               |

| `df.shape`             | Find the number of rows and columns        |

| `df.columns`           | View column names                          |

| `df.dtypes`            | Check data types                           |

| `df.info()`            | View dataset structure and non-null values |

| `df.describe()`        | Generate statistical summaries             |

| `df.isnull()`          | Identify missing values                    |

| `df.isnull().sum()`    | Count missing values                       |

| `df.fillna()`          | Replace missing values                     |

| `df.mean()`            | Calculate the mean of numerical data       |

| `df.duplicated()`      | Identify duplicate records                 |

| `df.drop\_duplicates()` | Remove duplicate records                   |

| `df.to\_csv()`          | Save the processed dataset                 |



\---



\## 🔍 Step 1 – Loading the Dataset



The dataset was loaded into a Pandas DataFrame using:



```python

import pandas as pd



df = pd.read\_csv("../dataset/student\_data.csv")

```



A \*\*DataFrame\*\* is the main two-dimensional data structure provided by Pandas and can be thought of as a table consisting of rows and columns.



\---



\## 🔎 Step 2 – Exploring the Dataset



Several Pandas functions were used to understand the dataset before performing any cleaning.



\### View the Dataset



```python

df

```



\### First Five Records



```python

df.head()

```



\### Last Five Records



```python

df.tail()

```



\### Dataset Dimensions



```python

df.shape

```



Initial dataset shape:



```text

(16, 7)

```



This means the dataset contained:



\* \*\*16 rows\*\*

\* \*\*7 columns\*\*



\### Column Names



```python

df.columns

```



\### Data Types



```python

df.dtypes

```



\### Dataset Information



```python

df.info()

```



\### Statistical Summary



```python

df.describe()

```



These operations helped understand the structure, data types, missing values, and statistical characteristics of the dataset before cleaning.



\---



\## 🧹 Step 3 – Data Cleaning



The raw dataset contained intentional data quality issues so that basic preprocessing techniques could be practiced.



\### Data Issues Identified



| Issue            | Column             | Count |

| ---------------- | ------------------ | ----: |

| Missing value    | `Study\_Hours`      |     1 |

| Missing value    | `Assignment\_Score` |     1 |

| Duplicate record | Complete row       |     1 |



\---



\## ❌ Handling Missing Values



Missing values were identified using:



```python

df.isnull()

```



The total number of missing values in each column was then calculated using:



```python

df.isnull().sum()

```



The missing values were handled by replacing them with the \*\*mean of their respective numerical columns\*\*.



\### Study Hours



```python

df\["Study\_Hours"] = df\["Study\_Hours"].fillna(

&#x20;   df\["Study\_Hours"].mean()

)

```



\### Assignment Score



```python

df\["Assignment\_Score"] = df\["Assignment\_Score"].fillna(

&#x20;   df\["Assignment\_Score"].mean()

)

```



This ensured that the dataset no longer contained missing values in these columns.



\---



\## 🔁 Handling Duplicate Records



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

df = df.drop\_duplicates()

```



The result was verified using:



```python

df.duplicated().sum()

```



After cleaning:



```text

0

```



duplicate records remained.



\---



\## ✅ Final Dataset



The dataset originally contained:



```text

16 rows × 7 columns

```



After removing the duplicate record:



```text

15 rows × 7 columns

```



The final dataset contained:



\* No missing values

\* No duplicate records

\* 15 valid student records

\* 7 columns



The cleaned dataset was saved as:



```text

cleaned\_student\_data.csv

```



\---



\## 💾 Saving the Cleaned Dataset



The cleaned DataFrame was exported using:



```python

df.to\_csv(

&#x20;   "../cleaned\_dataset/cleaned\_student\_data.csv",

&#x20;   index=False

)

```



The `index=False` parameter prevents Pandas from adding the DataFrame index as an unnecessary column in the output CSV file.



\---



\## 🐍 Python Script



Along with the Jupyter Notebook implementation, the complete data cleaning workflow was also implemented as a Python script:



```text

src/data\_cleaning.py

```



The script performs the complete process:



```text

Load Dataset

&#x20;    ↓

Explore Dataset

&#x20;    ↓

Check Missing Values

&#x20;    ↓

Fill Missing Values

&#x20;    ↓

Detect Duplicates

&#x20;    ↓

Remove Duplicates

&#x20;    ↓

Save Cleaned Dataset

```



This provides both an interactive \*\*Jupyter Notebook\*\* implementation and a reusable \*\*Python script\*\*.



\---



\## 📓 Jupyter Notebook



The complete interactive implementation is available in:



```text

notebooks/Data\_Cleaning.ipynb

```



The notebook contains the step-by-step exploration and cleaning process along with the corresponding outputs.



\---



\## 🛠️ Technologies Used



| Technology       | Purpose                        |

| ---------------- | ------------------------------ |

| Python           | Programming language           |

| Pandas           | Data manipulation and cleaning |

| Jupyter Notebook | Interactive development        |

| Git              | Version control                |

| GitHub           | Repository management          |



\---



\## 📚 Skills Learned



During Week 1, the following skills were developed:



\* Understanding the basics of Pandas

\* Creating and working with DataFrames

\* Loading CSV datasets

\* Exploring dataset structure

\* Understanding rows and columns

\* Identifying numerical and categorical data

\* Understanding features and labels

\* Detecting missing values

\* Handling missing values using mean imputation

\* Detecting duplicate records

\* Removing duplicate records

\* Verifying cleaned data

\* Exporting cleaned datasets

\* Creating reusable Python data-processing scripts

\* Organizing a data science project

\* Using Git and GitHub for version control



\---



\## 🧠 Key Learning



The major learning from Week 1 was understanding that \*\*data preprocessing is an essential step before applying Machine Learning algorithms\*\*.



Raw data may contain missing values, duplicate records, inconsistent information, or other quality issues. Pandas provides simple and powerful tools to identify and handle these problems efficiently.



The workflow learned in this project forms the foundation for the Machine Learning concepts that will be implemented in the upcoming weeks.



\---



\## 📈 Week 1 Outcome



At the end of Week 1, a raw student performance dataset was successfully loaded, explored, cleaned, verified, and exported into a separate cleaned dataset using Pandas.



The project successfully demonstrates the basic data preprocessing workflow required before moving toward feature engineering and Machine Learning model development.



\---



\## 🔗 Part of EDP Internship



This project is part of the \*\*EDP Internship\*\* and represents the work completed during \*\*Week 1\*\*.



Future weeks will build upon the concepts learned here and gradually introduce more advanced Data Science and Machine Learning techniques.



\---



\## 👨‍💻 Author



\*\*T. Navaneeth Reddy\*\*



B.Tech – Information Technology

Institute of Aeronautical Engineering



GitHub: \[Navaneeth4141](https://github.com/Navaneeth4141)



