from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from modules.inference_pipeline import LoanApprovalInferencePipeline
import uvicorn

app = FastAPI(title="Loan Approval API", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize pipeline
pipeline = LoanApprovalInferencePipeline(
    r'/Users/melodie.ezeani/Documents/loan_api_project/model/best_rf.pkl',
    r'/Users/melodie.ezeani/Documents/loan_api_project/model/feature_names.pkl'
)

class LoanRequest(BaseModel):
    annual_income: float
    dti: float
    loan_amount: int
    int_rate: float
    installment: float
    grade: str
    sub_grade: str
    emp_length: str
    home_ownership: str
    verification_status: str
    purpose: str
    term: str
    total_acc: int
    address_state: str

class LoanResponse(BaseModel):
    prediction: int
    probability_good_loan: float
    probability_bad_loan: float
    loan_decision: str
    risk_level: str

@app.get("/")
def read_root():
    return FileResponse('static/index.html')

@app.post("/predict", response_model=LoanResponse)
def predict_loan(request: LoanRequest):
    try:
        # Convert request to dict
        input_data = request.dict()
        
        # Make prediction
        result = pipeline.predict(input_data)
        
        return LoanResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)