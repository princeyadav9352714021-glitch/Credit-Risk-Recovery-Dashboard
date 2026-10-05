# Credit Risk Recovery Dashboard

## Overview
Credit Risk Recovery Dashboard is a Machine Learning-powered web application built using Streamlit. The project helps financial institutions analyze borrower risk, monitor loan recovery performance, and predict high-risk borrowers using a trained Random Forest model.

## Features

- Interactive Dashboard
- Credit Risk Prediction
- Borrower Portfolio Analysis
- Loan Recovery Monitoring
- CSV Dataset Upload
- Risk Score Calculation
- Recovery Strategy Recommendation
- Plotly Interactive Visualizations
- Dark Mode Interface
- Machine Learning Model Integration

## Technology Stack

- Python
- Streamlit
- Pandas
- Scikit-Learn
- Joblib
- Plotly

## Project Structure

```text
project/
│
├── app.py
├── credit_score_model.pkl
├── scaler.pkl
├── loan-recovery.csv
├── requirements.txt
└── README.md
```

## Dataset Features

The model uses the following borrower attributes:

- Age
- Monthly Income
- Loan Amount
- Loan Tenure
- Interest Rate
- Collateral Value
- Outstanding Loan Amount
- Monthly EMI
- Number of Missed Payments
- Days Past Due

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd project-folder
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

## Model Information

Algorithm Used:

- Random Forest Classifier

The model predicts the probability of borrower default and categorizes customers into:

- Low Risk
- Medium Risk
- High Risk

## Risk Strategy

### Low Risk
- Automated reminders
- Regular monitoring

### Medium Risk
- Settlement offers
- Flexible repayment plans

### High Risk
- Immediate recovery action
- Legal notices
- Aggressive collection strategy

## Dashboard Components

### Portfolio Overview
Displays:
- Total Accounts
- Recovered Accounts
- At-Risk Accounts
- Dataset Statistics

### Visual Analytics
- Loan Amount Distribution
- Monthly Income vs Loan Amount
- Payment History Analysis
- Recovery Status Insights

### Prediction Module
Users can enter borrower information and receive:
- Risk Score
- Risk Level
- Recovery Recommendation

## Deployment

### Streamlit Cloud

1. Upload project to GitHub.
2. Connect GitHub repository with Streamlit Cloud.
3. Select app.py as entry point.
4. Deploy.

## Future Improvements

- Real-time data integration
- SHAP Explainable AI
- Advanced borrower segmentation
- PDF report generation
- Email alert system
- Loan default forecasting

## Author

Prince Yadav

Data Science Trainee
TIPS G Alwar

## License

This project is developed for educational and portfolio purposes.
