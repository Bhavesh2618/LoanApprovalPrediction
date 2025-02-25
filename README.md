# Loan Approval Prediction using ML

## Overview
This project focuses on predicting loan approvals using various machine learning algorithms. It involves data preprocessing, visualization, and model training with accuracy comparisons. Additionally, a custom algorithm was created to predict loan approvals by selecting the best-performing model.

## 📊 Features
- **Data Cleaning and Imputation:** Handles missing values for features like Dependents, LoanAmount, Loan_Amount_Term, and Credit_History.
- **Visualization:** Histogram, scatter plots, and box plots for data insights.
- **Model Training:** Implements KNN, SVM, Logistic Regression, Random Forest, and Decision Tree classifiers.
- **Accuracy Comparison:** Evaluates model performance using accuracy scores.
- **Custom Algorithm:** Predicts loan approvals using the top-performing algorithm.

## 🚀 Technologies Used
- **Pandas:** Data manipulation and analysis.
- **NumPy:** Numerical computing.
- **Matplotlib & Seaborn:** Data visualization.
- **Scikit-learn:** Machine learning algorithms and model evaluation.

## 📁 Folder Structure
```
/LoanApprovalPrediction.csv  # Dataset
/main.py                     # Main code file
/README.md                   # Project documentation
```

## 📈 Models Implemented
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Random Forest Classifier**
- **Decision Tree Classifier**

## 📊 Model Accuracy Comparison
The project compares the accuracy of different models and visualizes the results using bar plots.

## 🎯 Custom Algorithm
After training and evaluating all models, the algorithm with the highest accuracy was used to create a new function that predicts loan approvals. The custom algorithm dynamically calls the top-performing model from the dataset to make predictions.

## 📧 Contact
- **Email:** bhavesh.kumarp2618@gmail.com
- **Phone:** 8522022445
- **LinkedIn:** [Bhavesh Kumar](www.linkedin.com/in/bhavesh-kumar2618)

## 🏗️ Setup
Follow these steps to run the project locally:

1. Clone this repository:
```bash
git clone https://github.com/Bhavesh2618/Loan-Approval-Prediction.git
```
2. Navigate to the project directory:
```bash
cd Loan-Approval-Prediction
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```
4. Run the main script:
```bash
python main.py
```

## 📜 License
This project is licensed under a **view-only** policy. Modifications and edits are not permitted.

---
Feel free to explore the project, visualize the data, and understand how loan approval predictions work using machine learning models!

