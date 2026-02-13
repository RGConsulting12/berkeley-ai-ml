# Practical Application III: Comparing Classifiers

## Project Overview

This project analyzes a Portuguese bank's telemarketing campaign data to predict whether a client will subscribe to a term deposit. We compare four classification algorithms to identify the best approach for targeting potential customers.

## Business Question

**"How can we predict which customers are most likely to subscribe to a term deposit, so we can prioritize our telemarketing calls?"**

### How We Answer This Question

1. **Collect customer data** (demographics, financial status, campaign history, economic conditions)
2. **Train a machine learning model** to learn patterns from 41,188 past customer interactions
3. **Score each customer** with a probability (0-100%) of subscribing
4. **Rank customers** by probability and call the highest-ranked first
5. **Measure impact**: Calling only the top 20% reaches ~55% of all potential subscribers

## Business Problem

The bank wants to optimize its telemarketing efforts by:
1. Identifying clients most likely to subscribe to term deposits
2. Reducing wasted calls to unlikely prospects
3. Improving campaign ROI through better targeting

## Dataset

- **Source**: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Size**: 41,188 records with 20 features
- **Target Variable**: Whether the client subscribed to a term deposit (yes/no)
- **Class Imbalance**: ~11% positive class (subscribed)

## Methodology

### Models Compared
1. **Logistic Regression** - Baseline interpretable model
2. **K-Nearest Neighbors (KNN)** - Distance-based classification
3. **Decision Tree** - Rule-based interpretable model
4. **Support Vector Machine (SVM)** - Margin-based classification

### Evaluation Metric
**ROC-AUC** was chosen as the primary metric because:
- The dataset is imbalanced (11% positive class)
- We need to balance precision and recall for business optimization
- AUC provides a threshold-independent measure of model quality

## Key Findings

### Model Performance Summary

| Model | ROC-AUC | Training Time | Key Insight |
|-------|---------|---------------|-------------|
| **Logistic Regression** | ~0.79 | Fast | Best balance of performance & interpretability |
| Decision Tree | ~0.72 | Medium | Prone to overfitting |
| KNN | ~0.75 | Medium | Struggles with high-dimensional data |
| SVM | ~0.78 | Slow | Good performance but computationally expensive |

### Top Predictive Features
1. **euribor3m** (3-month Euribor rate) - Economic indicator
2. **nr.employed** (Number of employees) - Economic indicator
3. **age** - Client demographic
4. **campaign** (Number of contacts) - Contact fatigue indicator

### Business Recommendations

1. **Target High-Propensity Clients First**: Use the model to rank prospects and call the top deciles first
2. **Monitor Economic Indicators**: Adjust campaign intensity based on euribor3m and employment rates
3. **Limit Contact Attempts**: Diminishing returns after 3-4 contacts per campaign
4. **Exclude `duration` from Predictions**: This variable is only known after the call ends (data leakage)

## Next Steps

1. **Threshold Optimization**: Tune decision threshold based on cost/benefit analysis
2. **Probability Calibration**: Ensure predicted probabilities match actual conversion rates
3. **Model Monitoring**: Track performance drift as economic conditions change
4. **A/B Testing**: Validate model lift in production environment

## Repository Structure

```
module17_starter/
├── README.md                    # This file
├── prompt_III.ipynb            # Main analysis notebook
├── CRISP-DM-BANK.pdf           # Reference paper
├── data/
│   ├── bank-additional-full.csv    # Full dataset
│   ├── bank-additional.csv         # Sample dataset
│   └── bank-additional-names.txt   # Feature descriptions
└── images/
    ├── model_comparison.png
    ├── best_model_evaluation.png
    └── logistic_regression_metrics.png
```

## How to Run

1. Open `prompt_III.ipynb` in Jupyter Notebook/Lab
2. Run all cells sequentially
3. Review the "Key Findings and Recommendations" section for results

## Author

UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence

## References

- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems.
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
