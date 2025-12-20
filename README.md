# Stay Ahead – Churn Prediction System
This repository contains a complete machine learning pipeline for customer churn prediction, developed as part of the Samsung Innovation Campus.

The goal of the project is to predict which customers are likely to churn by experimenting with range of models (Traditional and Deep models), understand the factors influencing churn, and support proactive business decisions such as targeted retention campaigns.

# Project Overview
Customer churn poses a major challenge for subscription-based services, as losing customers is significantly more costly than retaining them. This project builds and evaluates multiple machine-learning models to classify customers into churn vs non-churn categories, with a focus on maximizing recall, because of the importance of correctly identifying customers who are likely to churn, since retaining an existing subscriber is much cheaper and more valuable than acquiring a new one.

We evaluate classical ML models as well as modern deep tabular models:
- Logistic Regression (Baseline)
- Random Forest
- XGBoost (Final Best Model)
- TabNet
- NODE (Neural Oblivious Decision Ensemble)

The objective is to identify high-risk customers early and enable businesses to take strategic action

## Dataset: Telco Customer Churn
We use the Telco Customer Churn dataset available on Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
The dataset includes 7,043 customers, with features covering:
- Demographics: gender, age, partner, dependents
- Services: phone, internet, online security, streaming, tech support
- Contracts: contract type, paperless billing, payment methods
- Financials: monthly charges, total charges
- Target Variable: Churn (Yes/No)

## Preprocessing:
- Removal of invalid TotalCharges entries
- Handling missing values
- One-Hot Encoding for categorical columns
- Standardization of numerical variables
- Train/test split (Stratified)
- Class balancing via:
-- class weights
-- threshold tuning

## Methodology
1. Baseline Model — Logistic Regression
- Simple, interpretable baseline
- Helps establish performance lower bound
- Performs reasonably but struggles with nonlinear patterns

2. Random Forest 
- Strong non-linear baseline
- Useful for capturing interactions between features
- Moderate performance but overfits without tuning
- Hyperparameters optimiztion using RandomSearch 

3. XGBoost – Final Selected Model
- Supervised gradient boosting algorithm
- Tuned using Bayesian Optimization (Optuna)
- Hyperparameters optimiztion using Basiyan optimization
- Achieved best overall recall and AUC performance

4. Deep Tabular Models
- TabNet
-- Sequential attention model
-- Provides interpretability via feature masks
-- Good performance but less stable than XGBoost

- NODE
-- Tree-inspired neural architecture
-- Tuned using Optuna
-- Strong results but computationally heavier
-- Hyperparameters optimiztion using Basiyan optimization

## Evaluation Strategy

To ensure robust assessment:
- 5-Fold Stratified Cross-Validation
- Recall used as the primary metric (priority: catch churn cases)

Additional metrics: 
- Accuracy
- Precision
- F1 score
- ROC-AUC

We also applied threshold tuning to shift model predictions toward higher recall when needed.

## Results Summary
1. Logistic Regression
Good interpretability, moderate performance, but struggles with non-linear churn drivers

2. Random Forest
Improved accuracy with high variance across folds

3. XGBoost – Final Best Model

Achived best Recall: ~80%
Strong ROC-AUC: ~0.84
Best trade-off between sensitivity and robustness

4. TabNet & NODE
Competitive results, and harder to tune. However, slightly lower performance than XGBoost in recall

Overall, XGBoost consistently delivered the most reliable performance.

## System Deployment

The final system includes:
- Backend (FastAPI)
/predict endpoint for real-time churn scoring
/high-risk-users to support dashboards

- Model serialized and served via REST
- Frontend (HTML/CSS/JS)
- Real-time churn prediction interface
- Risk users dashboard
- English & Arabic support

## Key Takeaways
- Churn prediction systems must focus on recall, not just accuracy
- Preprocessing decisions (encoding, scaling, balancing) significantly impacted results
- XGBoost provided the strongest performance after hyperparameter tuning
- Deep models (TabNet, NODE) are promising but require careful optimization
- Threshold tuning improved churn detection in the minority class
