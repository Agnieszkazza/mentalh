# Mental Health Diagnosis and Treatment Analysis 🧠

## Overview
This project analyzes mental health diagnosis and treatment data using Python and SQLite. The goal is to explore relationships between various factors, calculate key metrics, and visualize insights using machine learning and statistical methods. Additionally, a Random Forest Classifier is applied to predict outcomes based on treatment effectiveness and other features.

---

## Project Features

### Data Preprocessing
- **Handling Missing Values**:
  - Numeric columns are filled with their mean.
  - Categorical columns are filled with their mode.
- **Data Scaling**:
  - Numeric columns are standardized using `StandardScaler`.

### SQLite Integration
- Data from a CSV file is imported into an SQLite database for structured querying.
- Tables are created to store aggregated data, including:
  - Averages and counts grouped by diagnosis, gender, medication, and therapy type.

### Exploratory Data Analysis (EDA)
- Visualizations highlight trends and relationships:
  - **Average symptom severity by diagnosis.**
  - **Gender distribution across diagnoses.**
  - **Medication and therapy types used for different diagnoses.**
- Pair plots showcase relationships between numerical variables.

### Feature Engineering
- New features computed for deeper insights:
  - `Treatment Effectiveness`, `Adherence Index`, and `Mood Improvement Ratio`.
- Categorical variables are encoded using `LabelEncoder`.

### Machine Learning
- A **Random Forest Classifier** predicts mental health outcomes using engineered features.
- **Feature Importance**:
  - Key features are ranked by their predictive power.
- **Model Evaluation**:
  - Classifier performance is assessed using a classification report.

### Correlation Analysis
- A heatmap of correlations between numerical variables identifies relationships and multicollinearity.

---

## Setup Instructions

### Prerequisites
- **Python 3.8 or later**
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `sqlite3`
  - `seaborn`
  - `matplotlib`
  - `scikit-learn`
  - `imbalanced-learn`
  - `xgboost`

### Steps to Run
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   pip install -r requirements.txt
   python analysis.py

