
# Manufacturing Quality Assurance 🏭

A comprehensive **Machine Learning project for manufacturing defect severity classification** using Python and Scikit-learn to analyze manufacturing quality data, engineer temporal features, develop multiple classification models, and investigate the factors influencing defect severity predictions.

The project focuses on classifying manufacturing defects into **Critical, Minor, and Moderate** severity levels, evaluating classification performance, identifying model limitations, and explaining predictions through feature importance analysis and SHAP.

---

## 🌟 Features

### Core Features

* **Data Understanding & Cleaning:** Inspection of dataset structure, data types, missing values, duplicates, and categorical distributions.
* **Exploratory Data Analysis:** Analysis of defect severity, defect types, inspection methods, defect locations, repair costs, and temporal patterns.
* **Data Preprocessing:** Stratified train-test splitting, categorical encoding, numerical scaling, and preprocessing pipelines.
* **Feature Engineering:** Extraction of temporal characteristics from defect inspection dates.
* **Multiple Machine Learning Models:** Development and comparison of traditional, ensemble-based, and gradient-boosting classifiers.
* **Hyperparameter Tuning:** Optimization of selected classification models using cross-validation.
* **Ensemble Learning:** Implementation of hard voting, soft voting, weighted voting, and stacking classifiers.
* **Experiment Tracking:** Tracking model configurations, evaluation metrics, and experiments using MLflow.
* **Model Evaluation:** Assessment using Accuracy, Macro Precision, Macro Recall, Macro F1, classification reports, and confusion matrices.
* **Critical Defect Analysis:** Investigation of Critical-class recall, precision, missed defects, and false alerts.
* **Model Explainability:** SVM coefficients, permutation importance, Random Forest feature importance, and cross-model comparisons.
* **SHAP Analysis:** Global, class-specific, and individual prediction explanations using SHAP.
* **Quality Assurance Insights:** Identification of predictive limitations and considerations for real-world manufacturing applications.

---

## 🗂 Project Structure

The project is organized as follows:

```text
Manufacturing Quality Assurance/
│
├── data/
│   ├── raw/                            # Original manufacturing defects dataset
│   └── processed/                      # Processed datasets, splits, and analysis results
│
├── notebooks/
│   ├── 01_data_understanding.ipynb     # Dataset inspection and understanding
│   ├── 02_eda.ipynb                    # Exploratory Data Analysis
│   ├── 03_data_preparation.ipynb       # Data preparation and preprocessing
│   ├── 04_feature_engineering.ipynb    # Temporal feature engineering
│   ├── 05_model_development.ipynb      # Classification models and hyperparameter tuning
│   ├── 06_model_evaluation.ipynb       # Model evaluation and Critical defect analysis
│   └── 07_model_explainability.ipynb   # Feature importance and SHAP analysis
│
├── models/
│   ├── preprocessed/                   # Preprocessing pipeline and feature metadata
│   └── trained/                        # Saved classification models
│
├── reports/
│   └── figures/                        # EDA, evaluation, and explainability figures
│
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── .gitignore                          # Git ignored files
```

---

## 🏗️ Project Architecture

```text
+-------------------------------+
|          DATA LAYER           |
|                               |
|  Manufacturing Defects Data   |
|  1,000 Observations           |
+---------------+---------------+
                |
                v
+-------------------------------+
|      DATA UNDERSTANDING       |
|                               |
|  - Dataset inspection         |
|  - Data quality               |
|  - Missing values             |
|  - Severity distribution      |
+---------------+---------------+
                |
                v
+-------------------------------+
|           EDA LAYER           |
|                               |
|  - Defect type analysis       |
|  - Severity analysis          |
|  - Inspection methods         |
|  - Defect locations           |
|  - Repair cost patterns       |
|  - Temporal analysis          |
+---------------+---------------+
                |
                v
+-------------------------------+
|      DATA PREPARATION         |
|                               |
|  - Feature selection          |
|  - Leakage prevention         |
|  - Stratified train-test split|
|  - Target encoding            |
+---------------+---------------+
                |
                v
+-------------------------------+
|      FEATURE ENGINEERING      |
|                               |
|  - Month                      |
|  - Day                        |
|  - Day of Week                |
|  - Is Weekend                 |
+---------------+---------------+
                |
                v
+-------------------------------+
|      PREPROCESSING LAYER      |
|                               |
|  - One-Hot Encoding           |
|  - StandardScaler             |
|  - ColumnTransformer          |
|  - Feature transformation     |
+---------------+---------------+
                |
                v
+-------------------------------+
|         MODELING LAYER        |
|                               |
|  - Multiple classifiers       |
|  - Hyperparameter tuning      |
|  - Cross-validation           |
|  - Voting ensembles           |
|  - Stacking ensemble          |
|  - MLflow tracking            |
+---------------+---------------+
                |
                v
+-------------------------------+
|        EVALUATION LAYER       |
|                               |
|  - Accuracy                   |
|  - Macro Precision            |
|  - Macro Recall               |
|  - Macro F1                   |
|  - Confusion matrices         |
|  - Critical defect analysis   |
|  - Generalization assessment  |
+---------------+---------------+
                |
                v
+-------------------------------+
|      EXPLAINABILITY LAYER     |
|                               |
|  - SVM coefficients           |
|  - Permutation importance     |
|  - Random Forest importance   |
|  - Cross-model comparison     |
|  - SHAP global explanations   |
|  - SHAP local explanations    |
+---------------+---------------+
                |
                v
+-------------------------------+
|    QUALITY ASSURANCE INSIGHTS |
|                               |
|  Defect severity patterns     |
|  Classification limitations   |
|  Critical defect detection    |
|  Model reliability            |
|  Future improvements          |
+-------------------------------+
```

---

## 🛠 Tech Stack

* **Language:** Python
* **Development Environment:** VS Code, Jupyter Notebook
* **Libraries:**
  * **Data Manipulation:** Pandas, NumPy
  * **Visualization:** Matplotlib, Seaborn
  * **Machine Learning:** Scikit-learn, XGBoost
  * **Model Explainability:** SHAP, Scikit-learn Inspection
  * **Experiment Tracking:** MLflow
  * **Model Persistence:** Joblib
* **Version Control:** Git & GitHub

---

## 📊 Dataset Overview

The project uses the **Manufacturing Defects Dataset** available on Kaggle, containing **1,000 observations and 8 original variables** related to manufacturing defects, inspection procedures, severity classifications, and repair costs.

**Dataset Source:** [Manufacturing Defects — Kaggle](https://www.kaggle.com/datasets/fahmidachowdhury/manufacturing-defects)

### Dataset Features

```text
defect_id
product_id
defect_type
defect_date
defect_location
severity
inspection_method
repair_cost
```

### Feature Descriptions

| Feature | Description |
|---|---|
| `defect_id` | Unique identifier for each defect record |
| `product_id` | Identifier of the associated product |
| `defect_type` | Category of manufacturing defect |
| `defect_date` | Date associated with the recorded defect |
| `defect_location` | Location of the defect within the product |
| `severity` | Defect severity classification |
| `inspection_method` | Method used to inspect or identify the defect |
| `repair_cost` | Recorded cost associated with defect repair |

### Target Variable

**`severity`**

The target contains three classes:

| Severity | Observations | Percentage |
|---|---:|---:|
| Minor | 358 | 35.8% |
| Critical | 333 | 33.3% |
| Moderate | 309 | 30.9% |

The objective is to predict defect severity using manufacturing defect characteristics, inspection methods, and engineered temporal features.

### Feature Selection

The following variables were excluded from model training:

* **`defect_id`:** Unique record identifier without meaningful predictive information.
* **`product_id`:** High-cardinality identifier excluded from the selected modeling approach.
* **`repair_cost`:** Excluded to avoid potential target leakage because repair costs may be determined after defect severity assessment.

The selected predictors were:

```text
defect_type
defect_location
inspection_method
defect_date
```

The date variable was subsequently transformed into engineered temporal features.

---

## 🎯 Problem Statement

Manufacturing organizations need reliable defect severity assessments to support quality control, inspection prioritization, repair planning, and operational decision-making.

Manual severity classification can be time-consuming and may introduce inconsistencies across inspection processes.

The objective of this project is to investigate whether machine learning can classify manufacturing defects into **Critical, Minor, and Moderate** severity levels using available defect characteristics, inspection information, and temporal features.

The project also evaluates the reliability of these predictions, with particular attention to identifying Critical defects and understanding the limitations of the available dataset.

---

## 🚀 Project Objectives

* Understand the distribution and characteristics of manufacturing defects.
* Analyze defect severity across different defect types, locations, and inspection methods.
* Investigate repair cost patterns and temporal characteristics through exploratory analysis.
* Prepare the dataset for supervised multiclass classification.
* Prevent target leakage by excluding inappropriate predictive variables.
* Engineer meaningful temporal features from defect dates.
* Develop and compare multiple machine learning classification models.
* Optimize selected classifiers through hyperparameter tuning.
* Investigate ensemble learning approaches.
* Track modeling experiments and evaluation metrics using MLflow.
* Evaluate classification performance using Macro F1 and other relevant metrics.
* Assess the detection of Critical defects and associated classification errors.
* Explain model predictions using feature importance and SHAP.
* Identify limitations and opportunities for improving manufacturing quality assurance predictions.

---

## 🔍 Exploratory Data Analysis

The EDA stage investigates the structure, distribution, and relationships within the manufacturing defects dataset.

### Analysis Areas

* Defect severity distribution
* Defect type distribution
* Defect location distribution
* Inspection method distribution
* Repair cost distribution
* Repair costs across severity levels
* Repair costs across defect types
* Severity distribution across defect categories
* Severity patterns across inspection methods
* Severity patterns across defect locations
* Temporal distribution of recorded defects
* Numerical relationships and variability

The exploratory analysis provided an understanding of manufacturing defect characteristics and informed subsequent feature selection and preprocessing decisions.

---

## ⚙️ Data Preparation & Feature Engineering

### Data Preparation

The dataset was prepared for supervised classification through the following steps:

1. Identifying the target variable.
2. Removing identifier variables and potential leakage features.
3. Separating predictors from the target.
4. Performing a stratified train-test split.
5. Encoding the target classes.
6. Identifying numerical, categorical, and binary features.
7. Preparing a reusable preprocessing pipeline.

### Train-Test Split

The dataset was divided into:

| Dataset | Observations | Percentage |
|---|---:|---:|
| Training | 800 | 80% |
| Testing | 200 | 20% |

A stratified split was used to preserve the approximate severity-class distribution across the training and testing datasets.

### Temporal Feature Engineering

The original `defect_date` variable was transformed into:

* `Month`
* `Day`
* `Day_of_Week`
* `Is_Weekend`

These features were introduced to investigate whether calendar-related characteristics contributed to defect severity classification.

### Feature Preprocessing

A Scikit-learn `ColumnTransformer` was used to apply:

* **One-Hot Encoding:** Transformation of categorical defect and inspection variables.
* **StandardScaler:** Standardization of numerical temporal features.
* **Binary Passthrough:** Retention of the weekend indicator.

The resulting processed dataset contained **13 features**.

Preprocessing was fitted using training data and then applied to the test data to avoid data leakage.

---

## 🤖 Machine Learning Approach

The project treats manufacturing defect severity prediction as a **supervised multiclass classification problem**.

### Modeling Workflow

```text
Data Understanding
        |
        v
Exploratory Data Analysis
        |
        v
Feature Selection
        |
        v
Stratified Train-Test Split
        |
        v
Feature Engineering
        |
        v
Preprocessing Pipeline
        |
        v
Baseline Classification
        |
        v
Multiple Classification Models
        |
        v
Hyperparameter Tuning
        |
        v
Ensemble Learning
        |
        v
Cross-Validation Comparison
        |
        v
Held-Out Test Evaluation
        |
        v
Model Explainability
```

### Classification Models

The project evaluated the following model families:

* Dummy Classifier
* Logistic Regression
* Decision Tree
* Random Forest
* Extra Trees
* Support Vector Machine
* K-Nearest Neighbors
* Gaussian Naive Bayes
* Gradient Boosting
* XGBoost

### Hyperparameter Tuning

Hyperparameter optimization was performed for selected models:

* Logistic Regression
* Random Forest
* Support Vector Machine
* K-Nearest Neighbors
* XGBoost

### Ensemble Learning

The following ensemble approaches were implemented:

* **Hard Voting:** Combines predictions through majority voting.
* **Soft Voting:** Combines predicted class probabilities.
* **Weighted Soft Voting:** Assigns different weights to participating classifiers.
* **Stacking:** Combines base-model predictions using a meta-classifier.

A total of **19 model configurations** were evaluated.

### Cross-Validation

Model development used **5-fold Stratified Cross-Validation**.

The primary model comparison metric was **Macro F1-score**, supported by Accuracy, Macro Precision, and Macro Recall.

Macro F1 was prioritized because it evaluates classification performance across all three severity classes without allowing the largest class to dominate the overall score.

### Experiment Tracking

MLflow was used to organize and track machine learning experiments, including model configurations and evaluation metrics.

Trained models were saved using Joblib for reuse during model evaluation and explainability analysis.

---

## 📈 Model Evaluation

Model performance was assessed using multiple classification metrics.

### Accuracy

Measures the proportion of correctly classified manufacturing defects.

### Macro Precision

Calculates the average precision across all three severity classes, giving each class equal weight.

### Macro Recall

Measures the average proportion of correctly identified observations across the severity classes.

### Macro F1-Score

Combines precision and recall into a balanced metric and averages the class-level F1-scores equally.

### Confusion Matrix

Examines correct and incorrect classifications across Critical, Minor, and Moderate defects.

### Critical Defect Analysis

Additional analysis focused on the Critical severity class, including:

* Critical Recall
* Critical Precision
* Critical F1-score
* Missed Critical defects
* False Critical alerts

---

### Cross-Validation Performance

The following models achieved the highest cross-validation Macro F1-scores:

| Model | CV Accuracy | CV Macro F1 |
|---|---:|---:|
| Tuned Support Vector Machine | 0.3750 | **0.3710** |
| Hard Voting Ensemble | 0.3688 | 0.3646 |
| Tuned Logistic Regression | 0.3638 | 0.3612 |
| Logistic Regression | 0.3613 | 0.3582 |
| Tuned Random Forest | 0.3700 | 0.3567 |
| Tuned K-Nearest Neighbors | 0.3688 | 0.3564 |

**Tuned Support Vector Machine** achieved the highest cross-validation Macro F1-score.

However, cross-validation performance remained relatively low across the evaluated classifiers.

### Held-Out Test Performance

The highest test Macro F1-scores were:

| Model | Test Accuracy | Test Macro F1 | Critical Recall |
|---|---:|---:|---:|
| Tuned K-Nearest Neighbors | 0.3500 | **0.3354** | 0.3939 |
| Support Vector Machine | 0.3550 | 0.3335 | 0.3636 |
| Extra Trees | 0.3350 | 0.3333 | 0.3333 |
| Random Forest | 0.3300 | 0.3236 | 0.3636 |
| Decision Tree | 0.3150 | 0.3150 | 0.3333 |
| XGBoost | 0.3200 | 0.3126 | 0.3030 |

Although Tuned K-Nearest Neighbors achieved the highest observed test Macro F1-score, this result was identified during final evaluation rather than used for additional model tuning.

### Generalization Analysis

The Tuned SVM achieved:

| Metric | Macro F1 |
|---|---:|
| Cross-Validation | 0.3710 |
| Test | 0.2679 |

The Random Forest achieved:

| Metric | Macro F1 |
|---|---:|
| Training | 0.9365 |
| Test | 0.3236 |

The substantial training-to-test performance gap for Random Forest indicates severe overfitting.

Overall, the models demonstrated **limited generalization**, suggesting that the available predictors do not provide sufficiently strong or reliable information for accurate defect severity classification.

### Critical Defect Detection

Critical defect identification was examined separately because failing to recognize a severe manufacturing defect may have significant operational consequences.

Key observations included:

* The original K-Nearest Neighbors model achieved the highest observed Critical Recall of **0.5000**, but generated 68 false Critical alerts.
* The Hard Voting Ensemble achieved Critical Recall of **0.4242** and Critical F1 of approximately **0.4029**.
* The Tuned SVM identified **26 of 66 Critical defects**, missing the remaining 40.

These findings demonstrate the trade-off between identifying Critical defects and generating false alerts.

No evaluated model demonstrated sufficient reliability for deployment in a safety-sensitive manufacturing quality assurance process.

---

## 🔎 Model Explainability

Model explainability was performed to understand how the trained classifiers used manufacturing defect characteristics and temporal features.

The analysis combined **model-specific explanations, model-agnostic feature importance, and SHAP-based explanations**.

### SVM Coefficient Analysis

The Tuned SVM used a linear kernel, allowing its pairwise classification coefficients to be examined.

The analysis investigated feature contributions across:

* Critical vs Minor
* Critical vs Moderate
* Minor vs Moderate

Weekend status showed relatively large coefficient magnitudes, particularly in the Critical vs Moderate comparison.

### Permutation Feature Importance

Permutation importance was used to measure how randomly shuffling features affected training Macro F1-score.

For Tuned SVM, `Is_Weekend` recorded the highest positive permutation importance.

### Grouped Original Feature Importance

One-hot encoded features were grouped by their original variables and permuted together.

This allowed the analysis to compare:

* Defect Type
* Defect Location
* Inspection Method
* Month
* Day
* Day of Week
* Is Weekend

The Tuned SVM emphasized weekend status, whereas Random Forest showed greater training-set reliance on several manufacturing defect characteristics and temporal variables.

### Random Forest Feature Importance

Random Forest was analyzed using:

* Impurity-based feature importance
* Individual-feature permutation importance
* Grouped original-feature permutation importance

The impurity-based analysis identified Day, Month, and Day of Week as the most important processed features.

Grouped permutation importance highlighted Defect Location, Inspection Method, and Defect Type.

### Cross-Model Feature Importance Comparison

Feature rankings were compared between Tuned SVM and Random Forest.

| Original Feature | Tuned SVM Rank | Random Forest Rank |
|---|---:|---:|
| Is Weekend | 1 | 7 |
| Inspection Method | 2 | 2 |
| Defect Type | 3 | 3 |
| Day | 4 | 4 |
| Day of Week | 5 | 6 |
| Month | 6 | 5 |
| Defect Location | 7 | 1 |

The results demonstrated that different algorithms relied on substantially different features.

---

## 🧠 SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to investigate Random Forest predictions in greater detail.

The analysis was performed on **100 sampled training observations** across all three severity classes.

### Global SHAP Feature Importance

The highest mean absolute SHAP values were:

| Rank | Feature | Mean Absolute SHAP |
|---:|---|---:|
| 1 | Day | 0.0478 |
| 2 | Day of Week | 0.0446 |
| 3 | Month | 0.0422 |
| 4 | Is Weekend | 0.0258 |
| 5 | Manual Testing | 0.0235 |
| 6 | Structural Defect Type | 0.0208 |

Temporal features showed the largest average contributions to the Random Forest's predicted severity probabilities.

### Class-Specific SHAP Analysis

SHAP importance was calculated separately for:

* Critical defects
* Minor defects
* Moderate defects

The analysis showed that temporal features remained influential across all three classes, although their relative importance varied.

### Individual Prediction Explanations

SHAP waterfall plots were generated for individual predictions belonging to each severity class.

These visualizations illustrated how individual features increased or decreased predicted class probabilities relative to their baseline values.

### Interpretation Considerations

SHAP explains the behavior of the trained classifier rather than establishing that its predictions are correct.

Because the Random Forest exhibited substantial overfitting, SHAP findings were interpreted cautiously and were not treated as evidence of reliable or causal manufacturing relationships.

---

## 💡 Key Findings

* The manufacturing defects dataset contained a relatively balanced distribution of Critical, Minor, and Moderate severity classes.
* Multiple classification algorithms and ensemble methods demonstrated limited predictive performance.
* Tuned SVM achieved the highest cross-validation Macro F1-score of **0.3710**.
* Tuned K-Nearest Neighbors achieved the highest observed test Macro F1-score of **0.3354**.
* Random Forest demonstrated substantial overfitting, achieving **0.9365 training Macro F1** compared with **0.3236 test Macro F1**.
* Critical defect detection remained challenging, with a considerable proportion of severe defects incorrectly classified.
* Tuned SVM and Random Forest exhibited different feature importance rankings.
* Random Forest's SHAP analysis emphasized temporal characteristics, particularly Day, Day of Week, and Month.
* Feature importance results were insufficient to establish reliable relationships between the available predictors and actual defect severity.
* The findings highlight the importance of data quality, meaningful predictors, robust validation, and model reliability in manufacturing quality assurance applications.

---

## ⚠️ Limitations & Future Improvements

### Current Limitations

**1. Limited Dataset Size**

The dataset contains only 1,000 observations, restricting the amount of information available for learning complex defect severity patterns.

**2. Synthetic Dataset**

The dataset is synthetic and may not accurately represent real manufacturing environments, inspection processes, or defect severity relationships.

**3. Limited Predictive Features**

The available predictors may not contain enough information to reliably distinguish Critical, Minor, and Moderate defects.

**4. Weak Model Generalization**

Several models demonstrated low test performance, while Random Forest exhibited substantial overfitting.

**5. Critical Defect Misclassification**

The evaluated classifiers frequently failed to identify Critical defects, limiting their suitability for operational quality assurance decisions.

**6. Explainability Limitations**

Permutation importance and SHAP analyses were performed using training observations. Their findings explain fitted model behavior rather than independently validated relationships.

### Future Improvements

* Collect larger and more representative manufacturing datasets.
* Incorporate defect measurements, sensor readings, production conditions, and equipment information.
* Include relevant manufacturing process and inspection history.
* Explore additional feature engineering approaches based on domain knowledge.
* Investigate cost-sensitive learning and class-specific decision thresholds.
* Improve Critical defect detection while controlling false alerts.
* Evaluate feature importance using held-out validation folds.
* Explore alternative classification architectures where appropriate.
* Assess model calibration and uncertainty estimation.
* Validate model performance using real-world manufacturing data.
* Introduce monitoring and retraining strategies if a sufficiently reliable model is developed.

---

## 📁 Notebook Workflow

The project is divided into seven notebooks to maintain a structured and reproducible workflow.

| Notebook | Purpose |
|---|---|
| `01_data_understanding.ipynb` | Dataset inspection, structure, and quality assessment |
| `02_eda.ipynb` | Exploratory Data Analysis and visualization |
| `03_data_preparation.ipynb` | Feature selection, splitting, and preprocessing |
| `04_feature_engineering.ipynb` | Temporal feature creation and transformation |
| `05_model_development.ipynb` | Classification models, tuning, ensembles, and MLflow tracking |
| `06_model_evaluation.ipynb` | Test evaluation, generalization, and Critical defect analysis |
| `07_model_explainability.ipynb` | Feature importance, cross-model comparisons, and SHAP |

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd "Manufacturing Quality Assurance"
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

#### Windows

```bash
.\.venv\Scripts\activate
```

#### Mac/Linux

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the `notebooks/` directory and run the notebooks in numerical order.

The notebooks use relative paths to access project datasets, saved models, and generated figures.

### 6. Exploring MLflow Experiments

The project uses MLflow for experiment tracking.

When the SQLite tracking database is available locally, the MLflow interface can be launched using:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the local MLflow URL displayed in the terminal to inspect the recorded experiments.

---

## 📌 Project Status

**Status:** Completed ✅

The project includes:

* Data understanding and quality assessment
* Exploratory Data Analysis
* Data preparation and preprocessing
* Temporal feature engineering
* Multiple classification models
* Hyperparameter tuning
* Ensemble learning
* MLflow experiment tracking
* Cross-validation and model comparison
* Held-out test evaluation
* Critical defect classification analysis
* Generalization and overfitting assessment
* SVM coefficient analysis
* Permutation feature importance
* Random Forest feature importance
* Cross-model explainability comparison
* Global and class-specific SHAP analysis
* Individual SHAP prediction explanations

**Deployment Status:** Research and portfolio project — not suitable for production deployment in its current form due to limited predictive reliability.

---

## 👨‍💻 Author

**Tejas Jadhav**

* GitHub: [@tejas-jadhav](https://github.com/TejasMJ)
* LinkedIn: [Tejas Jadhav](https://www.linkedin.com/in/tejas-m-jadhav/)
