# Loan Approval Model - Project Report

## Executive Summary

This project developed a machine learning-powered loan approval system using Random Forest classification. The system processes loan applications and provides approval/rejection decisions with risk assessments through a REST API and web interface.

## 1. Model Accuracy and Insights

### Model Performance
The project evaluated **Logistic Regression** and **Random Forest** algorithms. The baseline Logistic Regression model struggled to perform effectively due to the highly imbalanced nature of the dataset (significantly higher proportion of "Fully Paid" loans vs. "Charged Off").

### Model Accuracy & Evaluation Metrics
**Model Selection**: The Random Forest Classifier was chosen as the final model over Logistic Regression. It proved more capable of handling the non-linear relationships in the data and provided better mechanisms (like class weighting) to address the severe class imbalance.

**Evaluation Metrics (Final Model Performance)**: The model prioritized identifying risky borrowers (Class 1), resulting in the following performance metrics on the test set:

**Recall (Risky Loans): 70%**

**Significance**: The model successfully identified 70% of the actual loan defaulters. This is a critical metric for risk management as missing a defaulter is costly.

**F1-Score (Risky Loans): 0.32**

**Significance**: This score reflects the trade-off between Recall and Precision. While Recall is high, Precision was low (20%), indicating the model is conservative and flags many "good" loans as "bad" to ensure it catches the defaulters.

**Overall Accuracy: 58%**

**Insight**: While seemingly low, simple accuracy is often misleading in imbalanced datasets (where predicting "Paid" for everyone would yield ~86% accuracy but catch 0 defaults). The lower accuracy here is a result of the model deliberately over-predicting risk to maximize Recall.

## 2. Top Features Influencing Decisions
Based on Feature Importance and SHAP analysis, the following features were the strongest predictors of loan repayment behavior:

- **Loan Term**: Shorter terms (36 months) strongly correlate with repayment
- **Residual Income**: Borrowers with higher disposable income after paying installments are safer bets
- **Debt-to-Income (DTI) & Loan-to-Income (LTI) Ratios**: High ratios were clear indicators of financial stress and default risk
- **Credit Grade**: The assigned sub-grade provided strong predictive power, summarizing the borrower's credit history
- **Annual Income**: Lower income bands (<$25k) were significantly associated with higher default rates

## 3. Deployment Steps & Challenges

### Deployment Strategy

- **Serialization**: The trained Random Forest model and feature list were saved as .pkl files using Python's pickle module
- **API Development**: A web application (likely using Flask or FastAPI) was structured to load these artifacts and expose a /predict endpoint
- **Hosting**: The application was deployed to a cloud environment using Digital Ocean droplets (virtual machines) or their App Platform for scalable hosting

### Challenges Faced

- **Class Imbalance**: The dataset heavily favored "Fully Paid" loans. Overcoming this required aggressive class weighting, which improved Recall but hurt Precision
- **Feature Engineering**: Raw data (like simple income) wasn't enough. Complex interaction features (e.g., combining income bins with loan ratios) were required to extract meaningful signals

## 4. Recommendations for Improvement
Since the model currently suffers from low Precision (0.20) and low overall Accuracy (58%), the following steps are recommended for the next iteration:

- **Model Selection (Gradient Boosting)**: Experiment with XGBoost or LightGBM
- **Threshold Tuning**: Currently, the classification uses a default threshold of 0.5. Analyzing the Precision-Recall Curve to find an optimal custom threshold could significantly reduce the number of "False Alarms" (good borrowers rejected) without sacrificing too much Recall
- **Data Enrichment**: The current features are largely financial ratios. Incorporating behavioral data (e.g., transaction history) or alternative data points could help distinguish between a "risky-looking" good borrower and an actual defaulter
- **Cost-Sensitive Learning**: Implement a custom loss function that specifically penalizes False Negatives (missing a default) differently than False Positives to align the model mathematically with the business's risk appetite