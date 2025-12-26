# UltraML - Automated Machine Learning Dashboard

UltraML is a Streamlit-based dashboard that allows users to upload a dataset, configure a target variable, and train/evaluate 12 different machine learning algorithms instantly.

## Features
- **Data Upload**: Support for CSV file uploads.
- **Automatic Task Detection**: Automatically detects if the task is **Classification** or **Regression**.
- **Model Selection**: Choose from 12 algorithms including:
  - Logistic/Linear Regression
  - Decision Trees & Random Forests
  - XGBoost, Gradient Boosting, AdaBoost
  - SVM, KNN, Naive Bayes
  - Lasso & Ridge Regression
- **Visualization**:
  - Interactive Confusion Matrix for Classification.
  - Actual vs Predicted Scatter Plots for Regression.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SreeshaNarasimhan/ML-ALG.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the dashboard using Streamlit:
```bash
streamlit run app.py
```
