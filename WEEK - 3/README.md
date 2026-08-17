<div align="center">

# Week 3 – Text Preprocessing, TF-IDF & Spam Classification

**EDP Internship · Foundation Phase**

A beginner-level Machine Learning assignment focused on understanding **text preprocessing**, converting text into numerical features using **TF-IDF**, training a **Multinomial Naive Bayes** classifier, evaluating spam detection performance, optimizing the classification threshold, and saving the trained model for future predictions.

</div>

---

## Week 3 Overview

| Item        | Details                                                    |
| :---------- | :--------------------------------------------------------- |
| Phase       | Foundation – Machine Learning Basics                       |
| Week        | 3 of 12                                                    |
| Topic       | Text Preprocessing, TF-IDF & Spam Classification           |
| Deliverable | Train and Evaluate a Text Classification Model             |
| Dataset     | UCI SMS Spam Collection                                    |
| Language    | Python                                                     |
| Libraries   | Pandas, NumPy, Matplotlib, Scikit-learn, Joblib            |
| Model       | Multinomial Naive Bayes                                    |
| Environment | Jupyter Notebook                                           |

### Objective

The objective of Week 3 was to understand the fundamentals of **Natural Language Processing (NLP)** and text classification by preprocessing SMS messages, converting text into numerical features using **TF-IDF**, training a **Multinomial Naive Bayes** classification model, evaluating its performance using classification metrics and a confusion matrix, analyzing classification errors, optimizing the spam decision threshold, and saving the final trained model for future use.

The workflow followed in this assignment was:

```text
Load Dataset
     ↓
Explore & Validate
     ↓
Clean & Preprocess Text
     ↓
Train/Test Split
     ↓
TF-IDF Feature Extraction
     ↓
Train Naive Bayes Model
     ↓
Evaluate Baseline
     ↓
Analyze Errors & Probabilities
     ↓
Optimize Decision Threshold
     ↓
Final Evaluation
     ↓
Custom SMS Classification
     ↓
Save Model
     ↓
Verify Saved Model
```

---

## Project Structure

```text
Week-3-Spam-Classifier/
│
├── dataset/
│   └── SMSSpamCollection
│
├── models/
│   ├── tfidf_vectorizer.joblib
│   ├── spam_classifier.joblib
│   └── decision_threshold.joblib
│
├── notebooks/
│   └── Spam_Classifier.ipynb
│
├── requirements.txt
└── README.md
```

---

## Dataset

The **UCI SMS Spam Collection** dataset was used for this assignment.

The original dataset contains **5,574 SMS messages** classified into two categories:

* `ham` – legitimate/non-spam message
* `spam` – unwanted or promotional message

During the data-cleaning process, duplicate and normalized duplicate messages were identified and handled before model training.

### Dataset Summary

| Dataset Stage | Records |
| :------------ | ------: |
| Original Dataset | 5,574 |
| After Exact Deduplication | 5,171 |
| After Normalized Cleanup | 5,131 |
| Training Data | 4,104 |
| Testing Data | 1,027 |

### Class Distribution

| Class | Records |
| :---- | ------: |
| Ham | 4,502 |
| Spam | 629 |
| Total | 5,131 |

---

# Dataset Exploration

## Loading the Dataset

The SMS dataset was loaded into a Pandas DataFrame by reading the tab-separated dataset file.

```python
file_path = "../dataset/SMSSpamCollection"
```

Each record contains a message label and the corresponding SMS text.

The resulting DataFrame contains:

| Column | Description |
| :----- | :---------- |
| `label` | Classification label (`ham` or `spam`) |
| `message` | Original SMS message |

---

## Exploring the Dataset

The following operations were used during the initial exploration and validation:

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

### Check Missing Values

```python
df.isnull().sum()
```

### Check Duplicate Records

```python
df.duplicated().sum()
```

These operations helped understand the structure, quality, and distribution of the SMS dataset before model development.

---

# Text Preprocessing

Raw text cannot be directly provided to a traditional Machine Learning classifier.

Therefore, the SMS messages were cleaned and normalized before converting them into numerical features.

The preprocessing workflow included:

```text
Original SMS
     ↓
Convert to Lowercase
     ↓
Handle Common Emoticons
     ↓
Remove Unnecessary Symbols
     ↓
Normalize Whitespace
     ↓
Cleaned SMS
```

---

## Preprocessing Operations

The preprocessing function performs the following operations:

### 1. Convert Text to Lowercase

Example:

```text
"Congratulations! YOU WON!"
```

becomes:

```text
"congratulations! you won!"
```

### 2. Handle Common Emoticons

Common emoticons such as:

```text
:)
:-)
;)
```

are converted into a meaningful token such as:

```text
smile
```

This prevents potentially useful information from being completely discarded during cleaning.

### 3. Remove Unnecessary Symbols

Punctuation and unnecessary symbols are removed while preserving letters, numbers, and whitespace.

### 4. Normalize Whitespace

Multiple spaces are converted into a single space.

## Example

Original message:

```text
Congratulations!!! You WON $1000. Claim NOW!!!
```

After preprocessing:

```text
congratulations you won 1000 claim now
```

---

# Data Cleaning & Validation

Before training the model, additional data-quality checks were performed.

The cleaning process included:

* Exact duplicate detection
* Normalized duplicate detection
* Empty-message checking
* Label validation
* Class distribution analysis
* Train/test overlap checking

The training and testing messages were explicitly checked for overlap to reduce the risk of data leakage.

The final dataset contained:

```text
5,131 messages
```

with:

```text
4,502 ham messages
629 spam messages
```

---

# Features & Target

Unlike the regression task in Week 2, Week 3 uses text data.

### Feature

The SMS message is the input feature.

```text
SMS Message → Input Feature
```

### Target

The label represents the class the model needs to predict.

```text
ham / spam → Target / Label
```

The feature and target were separated as:

```python
X = df["clean_message"]
y = df["label"]
```

---

# Train/Test Split

The cleaned dataset was divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The final split contained:

| Dataset | Records |
| :------ | ------: |
| Training Data | 4,104 |
| Testing Data | 1,027 |
| Total | 5,131 |

### Why Train/Test Split?

The training data is used to allow the Machine Learning model to learn patterns from the SMS messages.

The testing data remains separate and is used to evaluate the model on messages that were not used during training.

A `random_state` of `42` was used to make the split reproducible.

The class distribution was also maintained using stratification.

---

# TF-IDF Feature Extraction

Machine Learning models require numerical input.

Since SMS messages are text, the cleaned messages were converted into numerical vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

The TF-IDF vectorizer was configured with:

```text
lowercase    : False
ngram_range : (1, 2)
```

The `lowercase=False` configuration was used because the text had already been converted to lowercase during preprocessing.

---

## Unigrams & Bigrams

The configuration:

```python
ngram_range=(1, 2)
```

allows the vectorizer to learn both:

### Unigrams

Individual words:

```text
free
winner
call
prize
```

### Bigrams

Two-word combinations:

```text
free entry
cash prize
call now
```

Using both unigrams and bigrams allows the model to capture individual words as well as short phrases.

---

## TF-IDF Features

The final TF-IDF representation contained:

```text
43,856 features
```

The training matrix therefore represents each SMS message as a numerical vector containing TF-IDF values.

The TF-IDF representation was fitted only on the training data and then used to transform the testing data.

This prevents information from the testing dataset from influencing the learned vocabulary.

---

# Multinomial Naive Bayes

A **Multinomial Naive Bayes** classifier was selected for the SMS spam classification task.

```python
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB(
    alpha=1.0
)

nb_model.fit(
    X_train_tfidf,
    y_train
)
```

### Model Configuration

| Parameter | Value |
| :-------- | :---- |
| Algorithm | Multinomial Naive Bayes |
| Alpha | 1.0 |
| Training Samples | 4,104 |
| TF-IDF Features | 43,856 |

Multinomial Naive Bayes is well suited for text classification because it works effectively with word-frequency and TF-IDF based representations.

---

# Baseline Model Evaluation

The initial Naive Bayes model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix

The baseline model achieved:

| Metric | Result |
| :----- | -----: |
| Accuracy | 93.18% |
| Precision | 100.00% |
| Recall | 44.44% |
| F1-Score | 61.54% |

### Baseline Confusion Matrix

```text
                Predicted
                Ham   Spam

Actual Ham       901     0
Actual Spam       70    56
```

The baseline model had very high precision but relatively low spam recall.

This means that while messages predicted as spam were highly reliable, a significant number of actual spam messages were still classified as ham.

Therefore, further analysis and threshold optimization were performed.

---

# Error Analysis

The incorrectly classified messages were analyzed to understand where the classifier was making mistakes.

Particular attention was given to:

* False positives
* False negatives
* Spam messages classified as ham
* Probability distributions
* Classification confidence

False negatives are especially important in spam detection because they represent spam messages that were incorrectly classified as legitimate messages.

This analysis helped motivate the decision-threshold optimization step.

---

# Decision Threshold Optimization

A standard binary classifier often uses a probability threshold of:

```text
0.50
```

However, spam detection may benefit from using a lower threshold because missing a spam message can be more undesirable than incorrectly flagging a legitimate message.

Therefore, multiple decision thresholds were evaluated.

The tested thresholds included:

```text
0.10
0.15
0.20
0.25
0.30
0.35
0.40
0.45
0.50
```

For each threshold, the following metrics were evaluated:

* Accuracy
* Precision
* Recall
* F1-Score

The primary optimization objective was to maximize the **spam F1-score**.

---

## Selected Decision Threshold

The final selected threshold was:

```text
0.10
```

This threshold was then used consistently by the final classifier.

```text
Spam probability ≥ 0.10
        ↓
      SPAM

Spam probability < 0.10
        ↓
       HAM
```

The selected threshold was also saved as a model artifact so that future predictions use the same decision rule.

---

# Final Model Evaluation

After selecting the optimized decision threshold, the final classifier was evaluated on the held-out testing dataset.

### Final Confusion Matrix

```text
                Predicted
                Ham   Spam

Actual Ham       896     5
Actual Spam       24   102
```

The final model produced the following approximate performance:

| Metric | Result |
| :----- | -----: |
| Accuracy | 97.18% |
| Precision | 95.33% |
| Recall | 80.95% |
| F1-Score | 87.18% |

The optimized threshold substantially improved spam recall and F1-score compared with the baseline model.

---

# Final Model Configuration

The final trained classifier uses:

| Component | Configuration |
| :--------- | :------------ |
| Text Representation | TF-IDF |
| N-gram Range | `(1, 2)` |
| TF-IDF Features | 43,856 |
| Classifier | Multinomial Naive Bayes |
| Alpha | 1.0 |
| Decision Threshold | 0.10 |

---

# Custom SMS Classifier

A reusable custom SMS classification function was created after training the final model.

The function performs the complete prediction pipeline:

```text
New SMS
   ↓
Preprocess Text
   ↓
TF-IDF Transform
   ↓
Naive Bayes Prediction
   ↓
Calculate Spam Probability
   ↓
Apply Decision Threshold
   ↓
HAM / SPAM
```

The classifier can accept a new SMS message and return:

* Original message
* Cleaned message
* Predicted label
* Spam probability

Example:

```text
Message:
Congratulations! You have won a cash prize.

Prediction:
SPAM
```

---

# Model Saving

The final trained model components were saved using **Joblib**.

The following files are generated:

| File | Purpose |
| :--- | :------ |
| `tfidf_vectorizer.joblib` | Stores the fitted TF-IDF vectorizer |
| `spam_classifier.joblib` | Stores the trained Multinomial Naive Bayes model |
| `decision_threshold.joblib` | Stores the selected classification threshold |

These files allow the trained model to be reused without retraining the entire pipeline.

---

# Model Verification & Reproducibility

After saving the model, all three artifacts were loaded again to verify that they could be successfully reused.

The saved configuration was verified as:

```text
TF-IDF features      : 43856
TF-IDF n-gram range  : (1, 2)
Naive Bayes alpha     : 1.0
Decision threshold    : 0.1
```

The saved model was also tested using new SMS messages.

Example results:

```text
Message 1:
Hey, I will reach home by 8 tonight.

Prediction:
HAM

Spam probability:
0.0037
```

```text
Message 2:
Congratulations! You have won a free cash prize. Call now!

Prediction:
SPAM

Spam probability:
0.7604
```

The final reproducibility test successfully passed:

```text
FINAL REPRODUCIBILITY TEST: PASSED
```

This confirms that the saved model artifacts reproduce the expected classifier configuration and can be loaded successfully for future predictions.

---

# Implementation

The assignment was implemented using a Jupyter Notebook.

| Implementation | File | Purpose |
| :-------------- | :--- | :------ |
| Jupyter Notebook | `notebooks/Spam_Classifier.ipynb` | Interactive learning, experimentation and model development |
| Saved Model | `models/` | Stores the trained classifier and supporting artifacts |

### Machine Learning Workflow

```text
SMS Spam Dataset
       ↓
Load Dataset
       ↓
Explore Dataset
       ↓
Clean & Validate Data
       ↓
Preprocess Text
       ↓
Train/Test Split
       ↓
TF-IDF Feature Extraction
       ↓
Train Multinomial Naive Bayes
       ↓
Baseline Evaluation
       ↓
Error Analysis
       ↓
Probability Analysis
       ↓
Decision Threshold Optimization
       ↓
Final Evaluation
       ↓
Custom SMS Classifier
       ↓
Save Model Components
       ↓
Verify Saved Model
```

---

## Jupyter Notebook

The complete interactive implementation is available in:

```text
notebooks/Spam_Classifier.ipynb
```

The notebook contains the step-by-step implementation of:

* Dataset loading
* Dataset exploration
* Data cleaning
* Text preprocessing
* Train/test splitting
* TF-IDF feature extraction
* Multinomial Naive Bayes training
* Model evaluation
* Error analysis
* Decision threshold optimization
* Final evaluation
* Custom SMS classification
* Model saving
* Reproducibility verification

---

# Technologies & Tools

| Technology / Tool | Purpose |
| :---------------- | :------ |
| Python | Primary programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Text feature extraction, Machine Learning and evaluation |
| Joblib | Saving and loading trained model artifacts |
| Jupyter Notebook | Interactive development and experimentation |
| Git | Version control |
| GitHub | Repository management |

---

# Running the Project

## Install Dependencies

Install the required libraries using:

```bash
pip install -r requirements.txt
```

## Run the Jupyter Notebook

Open:

```text
notebooks/Spam_Classifier.ipynb
```

and execute the cells sequentially.

The notebook will:

1. Load the SMS Spam Collection dataset
2. Explore and validate the dataset
3. Clean and preprocess the SMS messages
4. Split the dataset into training and testing sets
5. Convert text into TF-IDF features
6. Train the Multinomial Naive Bayes classifier
7. Evaluate the baseline model
8. Analyze classification errors
9. Optimize the decision threshold
10. Evaluate the final classifier
11. Test custom SMS messages
12. Save the trained model components
13. Verify the saved model for reproducibility

---

# Skills Learned

By completing Week 3, the following skills were developed:

* Understanding the fundamentals of text classification
* Understanding basic NLP preprocessing
* Cleaning and normalizing text data
* Handling duplicate and normalized duplicate records
* Identifying and preventing data leakage
* Understanding features and target variables in text classification
* Splitting text datasets into training and testing sets
* Understanding TF-IDF feature extraction
* Understanding unigrams and bigrams
* Converting text into numerical feature vectors
* Training a Multinomial Naive Bayes classifier
* Evaluating classification models
* Understanding accuracy, precision, recall and F1-score
* Interpreting confusion matrices
* Performing error analysis
* Understanding false positives and false negatives
* Working with prediction probabilities
* Optimizing classification decision thresholds
* Building a reusable text classification function
* Saving trained Machine Learning models using Joblib
* Loading and verifying saved model artifacts
* Performing reproducibility checks
* Organizing a Machine Learning project
* Using Git and GitHub for version control

---

# Key Learning

The major takeaway from Week 3 was understanding how **text data can be converted into numerical representations and used to train a Machine Learning classification model**.

The assignment demonstrated that text classification involves more than simply training a classifier. The text must first be cleaned and normalized, the dataset must be validated, and the messages must be converted into numerical features using a technique such as **TF-IDF**.

The assignment also demonstrated how **Multinomial Naive Bayes** can be used effectively for text classification and how evaluation metrics such as **precision, recall, and F1-score** provide different perspectives on model performance.

An important practical learning from this assignment was that the default probability threshold is not always optimal. By evaluating multiple thresholds, the spam recall and F1-score were improved while maintaining strong overall classification performance.

---

# Week 3 Outcome

The **Text Preprocessing, TF-IDF & Spam Classification** assignment was successfully completed.

The SMS Spam Collection dataset was:

**Loaded → Explored → Cleaned → Preprocessed → Split → Vectorized → Trained → Evaluated → Optimized → Saved → Verified**

A Multinomial Naive Bayes classifier was successfully trained using **4,104 training messages** and evaluated on **1,027 unseen testing messages**.

The final model used:

```text
TF-IDF Features      : 43,856
N-gram Range         : (1, 2)
Algorithm            : Multinomial Naive Bayes
Alpha                : 1.0
Decision Threshold   : 0.10
```

The final model achieved approximately:

```text
Accuracy  : 97.18%
Precision : 95.33%
Recall    : 80.95%
F1-Score  : 87.18%
```

The final classifier was successfully saved and reloaded, and the reproducibility verification passed successfully.

Week 3 established the foundation for **Natural Language Processing, text feature extraction, binary classification, model evaluation, and model persistence**, preparing for more advanced Machine Learning concepts in the upcoming weeks.

---

## Author

**T. Navaneeth Reddy**

B.Tech – Information Technology

Institute of Aeronautical Engineering

**GitHub:** [Navaneeth4141](https://github.com/Navaneeth4141)

---

<div align="center">

### EDP Internship · Week 3

**Learn • Implement • Analyze • Document**

</div>
