# Loan Approval Prediction API

A machine learning-powered REST API that predicts loan approval decisions using Random Forest classification. The system processes loan applications and returns approval/rejection decisions with risk assessments.

## Features

- **ML-Powered Predictions**: Random Forest model trained on loan data
- **REST API**: FastAPI-based web service
- **Web Interface**: Modern HTML form for easy loan application submission
- **Risk Assessment**: Categorizes loans as Low, Medium, or High risk
- **Comprehensive Logging**: Request/response tracking and error monitoring
- **Docker Support**: Containerized deployment ready

## Project Structure

```
loan_api_project/
├── app.py                    # FastAPI application
├── jupyter_notebooks/
│   └── Data Cleaning & EDA.ipynb
│   └──Feature_Engineering.ipynb
│   └── Modelling.ipynb
├── modules/
│   └── inference_pipeline.py # ML preprocessing and prediction pipeline
├── model/
│   ├── best_rf.pkl          # Trained Random Forest model
│   └── feature_names.pkl    # Feature alignment for inference
├── static/
│   └── index.html           # Web interface
├── data/                    # Training/test datasets
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
└── README.md               # This file
```

## Quick Start

### Local Development

1. **Clone and setup**:
```bash
git clone <repository-url>
cd loan_api_project
pip install -r requirements.txt
```

2. **Run the API**:
```bash
python app.py
```

3. **Access the application**:
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Docker Deployment

1. **Build and run**:
```bash
docker build -t loan-api .
docker run -p 8000:8000 loan-api
```

## API Usage

### Prediction Endpoint

**POST** `/predict`

**Request Body**:
```json
{
  "annual_income": 50000,
  "dti": 0.15,
  "loan_amount": 10000,
  "int_rate": 0.12,
  "installment": 300,
  "grade": "B",
  "sub_grade": "B2",
  "emp_length": "5 years",
  "home_ownership": "RENT",
  "verification_status": "Verified",
  "purpose": "debt_consolidation",
  "term": "36 months",
  "total_acc": 15,
  "address_state": "CA"
}
```

**Response**:
```json
{
  "prediction": 0,
  "probability_good_loan": 0.85,
  "probability_bad_loan": 0.15,
  "loan_decision": "APPROVED",
  "risk_level": "Low"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `annual_income` | float | Annual income in USD |
| `dti` | float | Debt-to-income ratio (0-1) |
| `loan_amount` | int | Requested loan amount |
| `int_rate` | float | Interest rate (0-1) |
| `installment` | float | Monthly payment amount |
| `grade` | string | Loan grade (A-G) |
| `sub_grade` | string | Loan sub-grade (A1-G5) |
| `emp_length` | string | Employment length |
| `home_ownership` | string | RENT/OWN/MORTGAGE/OTHER |
| `verification_status` | string | Verified/Source Verified/Not Verified |
| `purpose` | string | Loan purpose |
| `term` | string | Loan term (36/60 months) |
| `total_acc` | int | Total credit accounts |
| `address_state` | string | US state code |

## Model Information

- **Algorithm**: Random Forest Classifier
- **Target**: 0 = Good Loan (Approved), 1 = Bad Loan (Rejected)
- **Features**: engineered features including ratios, binned categories, and one-hot encoded variables
- **Risk Levels**: 
  - Low: probability_bad_loan ≤ 0.3
  - Medium: 0.3 < probability_bad_loan ≤ 0.7
  - High: probability_bad_loan > 0.7

## Development

### Adding Features

1. Modify `modules/inference_pipeline.py`
2. Update feature engineering in `create_advanced_features()`
3. Retrain model and update `model/` files

### Testing

```bash
# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### Logging

Logs are written to:
- Console output
- `loan_api.log` file

Log levels: INFO (default), DEBUG (detailed processing)

## Deployment

### Environment Variables

- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

### Production Considerations

- Use reverse proxy (nginx)
- Enable HTTPS
- Set up log rotation
- Monitor API performance
- Scale with load balancer

## Dependencies

- **FastAPI**: Web framework
- **pandas**: Data processing
- **scikit-learn**: ML model
- **numpy**: Numerical computing
- **uvicorn**: ASGI server
- **joblib**: Model serialization