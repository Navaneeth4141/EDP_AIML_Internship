<div align="center">

# EDP Internship

A structured collection of all tasks, assignments, projects, and learning outcomes completed during my **EDP Internship**.

This repository documents my journey from **Machine Learning fundamentals** to building a **Personal Finance Assistant** and finally developing an **AI Resume Builder** as the capstone project.

</div>

---

## About

This repository contains the complete work carried out during the **12-week EDP Internship**.

The internship follows a progressive approach, beginning with Machine Learning foundations, moving into a practical **Personal Finance Assistant**, and concluding with the development and deployment of an **AI Resume Builder**.

Each week is maintained in its own folder with the relevant source code, datasets, notebooks, documentation, and project files.

---

## Internship Roadmap

|    Phase   |  Weeks | Focus                      |
| :--------: | :----: | :------------------------- |
| Foundation |  1 – 3 | Machine Learning Basics    |
|   Project  |  4 – 8 | Personal Finance Assistant |
|  Capstone  | 9 – 12 | AI Resume Builder          |

---

## Internship Progress

|   Week  | Learning Topic                                           | Weekly Deliverable                  |    Status   |
| :-----: | :------------------------------------------------------- | :---------------------------------- | :---------: |
|  Week 1 | Pandas Basics, Features & Labels                         | Load and Clean a Sample Dataset     | ✅ Completed |
|  Week 2 | Linear Regression, Train/Test Split & Evaluation Metrics | Train First Regression Model        | ✅ Completed |
|  Week 3 | Text Preprocessing, TF-IDF & Naive Bayes                 | Build a Spam Classifier             | ✅ Completed |
|  Week 4 | Transaction Categorization                               | Build Categorization Model          |  ⏳ Upcoming |
|  Week 5 | Trend Visualization                                      | Add Spending Trend Charts           |  ⏳ Upcoming |
|  Week 6 | LLM API Integration                                      | Add AI Saving Tips                  |  ⏳ Upcoming |
|  Week 7 | LLM API Integration                                      | Add AI Saving Tips                  |  ⏳ Upcoming |
|  Week 8 | Documentation & GitHub                                   | GitHub + README                     |  ⏳ Upcoming |
|  Week 9 | Finalize Project Direction                               | Confirm Capstone Project Scope      |  ⏳ Upcoming |
| Week 10 | Core Project Feature Development                         | Working Prototype                   |  ⏳ Upcoming |
| Week 11 | Deployment                                               | Live Application                    |  ⏳ Upcoming |
| Week 12 | Final Polish & Demo Preparation                          | GitHub README + 5-Minute Demo Pitch |  ⏳ Upcoming |

---

## Week 1 Overview

### Pandas Basics & Data Cleaning

Week 1 focused on building the foundation required for working with datasets in Python.

The main concepts covered were:

* Pandas fundamentals
* DataFrames and CSV files
* Dataset exploration
* Features and Labels
* Missing value detection and handling
* Duplicate detection and removal
* Saving cleaned datasets

### Weekly Deliverable

**Load and Clean a Sample Dataset**

A Student Performance Dataset was loaded using Pandas, explored, cleaned, and exported as a separate cleaned dataset.

Detailed implementation and documentation are available in:

```text
Week-1-Pandas-Basics/
```

---

## Week 2 Overview

### Linear Regression & Model Evaluation

Week 2 focused on building the first Machine Learning regression model and understanding the basic workflow of training and evaluating a model.

The main concepts covered were:

* Linear Regression
* Features and Target Variables
* Train/Test Split
* Model Training
* Model Coefficient and Intercept
* Predictions
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score
* Regression Visualization

### Weekly Deliverable

**Train First Regression Model**

A Salary Prediction Dataset was created and used to train a Linear Regression model that predicts salary based on years of experience.

The dataset was divided into **80% training data and 20% testing data**. The trained model was evaluated using MAE, MSE, RMSE, and R².

### Model Results

| Metric |    Result |
| :----- | --------: |
| MAE    |    419.46 |
| MSE    | 223215.72 |
| RMSE   |    472.46 |
| R²     |    0.9993 |

The model achieved an R² score of **0.9993**, indicating that approximately **99.93% of the variation in salary** was explained by the model on the test dataset.

Detailed implementation and documentation are available in:

```text
Week-2-Linear-Regression/
```

---

## Week 3 Overview

### Text Preprocessing, TF-IDF & Spam Classification

Week 3 focused on understanding how text data can be cleaned, transformed into numerical features, and used to train a Machine Learning classification model.

The main concepts covered were:

* Text preprocessing and normalization
* Handling emoticons and unnecessary symbols
* Duplicate and normalized duplicate detection
* Data validation and leakage checking
* Train/Test Split
* TF-IDF feature extraction
* Unigrams and Bigrams
* Multinomial Naive Bayes
* Accuracy, Precision, Recall and F1-Score
* Confusion Matrix
* Error Analysis
* Prediction Probability Analysis
* Decision Threshold Optimization
* Custom SMS Classification
* Model Saving using Joblib
* Model Loading and Reproducibility Verification

### Weekly Deliverable

**Build a Spam Classifier**

The UCI SMS Spam Collection dataset was cleaned and preprocessed before being divided into training and testing data. TF-IDF was used to convert the SMS messages into numerical features, and a Multinomial Naive Bayes classifier was trained to distinguish between **ham** and **spam** messages.

The final model used **43,856 TF-IDF features**, an n-gram range of **(1, 2)**, a Naive Bayes `alpha` value of **1.0**, and a decision threshold of **0.10**.

### Model Results

| Metric | Result |
| :----- | -----: |
| Accuracy | 97.18% |
| Precision | 95.33% |
| Recall | 80.95% |
| F1-Score | 87.18% |

The trained model was saved using Joblib and successfully reloaded and tested. The final reproducibility verification passed successfully.

Detailed implementation and documentation are available in:

```text
Week-3-Spam-Classifier/
```

---

## Phase 1 – Foundations

### Weeks 1 – 3

The first phase focuses on developing the fundamental Machine Learning skills required for the upcoming projects.

```text
Week 1
Pandas + Features & Labels
        ↓
Week 2
Linear Regression + Model Evaluation
        ↓
Week 3
Text Preprocessing + TF-IDF + Naive Bayes
```

### Foundation Deliverables

| Week | Deliverable                     | Status |
| :--: | :------------------------------ | :----: |
|   1  | Load and Clean a Sample Dataset |    ✅   |
|   2  | Train First Regression Model    |    ✅   |
|   3  | Build a Spam Classifier         |    ✅   |

---

## Phase 2 – Personal Finance Assistant

### Weeks 4 – 8

The second phase focuses on building a practical **Personal Finance Assistant** by combining Machine Learning, data visualization, and LLM-based features.

```text
Transaction Categorization
        ↓
Spending Trend Visualization
        ↓
LLM Integration
        ↓
AI Saving Tips
        ↓
Documentation & GitHub
```

### Project Deliverables

|  Week | Deliverable                |
| :---: | :------------------------- |
|   4   | Build Categorization Model |
|   5   | Add Spending Trend Charts  |
| 6 – 7 | Add AI Saving Tips         |
|   8   | GitHub + README            |

---

## Phase 3 – Capstone Project

### Weeks 9 – 12

The final phase of the internship is dedicated to building the **AI Resume Builder**, which serves as the capstone project for the internship.

The project will progress from defining the scope to developing, deploying, and presenting a working application.

```text
Project Scope
      ↓
Working Prototype
      ↓
Deployment
      ↓
Final Polish
      ↓
Demo & Presentation
```

### Capstone Deliverables

| Week | Deliverable                               |
| :--: | :---------------------------------------- |
|   9  | Confirm the Scope of the Capstone Project |
|  10  | Working Prototype                         |
|  11  | Live Application                          |
|  12  | GitHub README + 5-Minute Demo Pitch       |

### Planned Deployment

The final application may be deployed using one of the following platforms:

* Render
* Railway
* Streamlit Cloud

The final deployment platform will be selected during the deployment phase.

---

## Repository Structure

```text
EDP-Internship/
│
├── Week-1-Pandas-Basics/
│   ├── dataset/
│   ├── cleaned_dataset/
│   ├── notebooks/
│   ├── src/
│   ├── requirements.txt
│   ├── .gitignore
│   └── README.md
│
├── Week-2-Linear-Regression/
│   ├── dataset/
│   ├── notebooks/
│   ├── outputs/
│   ├── src/
│   ├── requirements.txt
│   └── README.md
│
├── Week-3-Spam-Classifier/
│   ├── dataset/
│   ├── models/
│   ├── notebooks/
│   ├── requirements.txt
│   └── README.md
│
├── Week-4-Transaction-Categorization/
│   └── README.md
│
├── Week-5-Spending-Trend-Charts/
│   └── README.md
│
├── Week-6-AI-Saving-Tips/
│   └── README.md
│
├── Week-7-LLM-Integration/
│   └── README.md
│
├── Week-8-Documentation/
│   └── README.md
│
├── Week-9-AI-Resume-Builder-Scope/
│   └── README.md
│
├── Week-10-AI-Resume-Builder-Prototype/
│   └── README.md
│
├── Week-11-AI-Resume-Builder-Deployment/
│   └── README.md
│
├── Week-12-Final-Demo/
│   └── README.md
│
└── README.md
```

> Folder names can be adjusted as each week's implementation is started, while maintaining the same overall organization.

---

## Technologies & Tools

The technologies used will evolve throughout the internship as new concepts and projects are introduced.

| Technology / Tool | Purpose                                    |
| :---------------- | :----------------------------------------- |
| Python            | Primary programming language               |
| Pandas            | Data manipulation and preprocessing        |
| NumPy             | Numerical calculations                     |
| Matplotlib        | Data visualization                         |
| Jupyter Notebook  | Data analysis and experimentation          |
| Scikit-learn      | Machine Learning algorithms and evaluation |
| Git               | Version control                            |
| GitHub            | Source code and project management         |
| LLM APIs          | AI-powered features                        |
| Streamlit         | Potential application deployment           |
| Render / Railway  | Potential application deployment           |

Additional libraries, APIs, and tools will be documented as they are introduced.

---

## Learning Journey

The internship follows a practical progression:

```text
Machine Learning Foundations
            ↓
Data Processing & Model Building
            ↓
Regression & Model Evaluation
            ↓
Practical Project Development
            ↓
LLM Integration
            ↓
AI Application Development
            ↓
Deployment
            ↓
Capstone Project
```

This approach allows each stage of the internship to build upon the knowledge and skills developed in the previous stage.

---

## Documentation

Each week contains its own README file with detailed information about the corresponding assignment.

Weekly documentation may include:

* Objectives
* Concepts learned
* Dataset information
* Implementation details
* Source code
* Results and outputs
* Skills acquired
* Key learnings
* Project outcomes

The main README provides the overall internship roadmap, while the individual weekly README files contain the detailed technical documentation.

---

## Version Control

Git and GitHub are used throughout the internship to track development and maintain a history of completed work.

Each week's work is added to the same repository using descriptive commits.

Example:

```text
Week 1 → Completed Pandas data cleaning
Week 2 → Added Linear Regression model and evaluation
Week 3 → Added spam classifier
Week 4 → Added transaction categorization
...
Week 12 → Finalized capstone and demo
```

---

## Overall Internship Goal

The overall goal of this internship is to develop practical skills in **Data Science, Machine Learning, and Artificial Intelligence** by progressing from fundamental concepts to real-world application development.

By the end of the internship, the repository will contain:

* Machine Learning foundation exercises
* A complete Personal Finance Assistant
* An AI-powered Resume Builder
* Source code and documentation
* Deployed application
* Final project presentation and demo materials

---

## Author

**T. Navaneeth Reddy**

B.Tech – Information Technology

Institute of Aeronautical Engineering

**GitHub:** [Navaneeth4141](https://github.com/Navaneeth4141)

---

<div align="center">

### EDP Internship

**Learn • Build • Deploy • Document**

</div>
