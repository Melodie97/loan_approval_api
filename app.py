from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from modules.inference_pipeline import LoanApprovalInferencePipeline
import uvicorn
import logging
import time
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('loan_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Loan Approval API", version="1.0.0")

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url} from {request.client.host}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize pipeline
try:
    logger.info("Initializing loan approval pipeline...")
    pipeline = LoanApprovalInferencePipeline(
        r'model/best_rf.pkl',
        r'model/feature_names.pkl'
    )
    logger.info("Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    raise

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
    logger.info("Serving main page")
    return FileResponse('static/index.html')

@app.post("/predict", response_model=LoanResponse)
def predict_loan(request: LoanRequest):
    try:
        logger.info(f"Prediction request received: loan_amount={request.loan_amount}, annual_income={request.annual_income}")
        
        # Convert request to dict
        input_data = request.dict()
        
        # Make prediction
        result = pipeline.predict(input_data)
        
        logger.info(f"Prediction completed: {result['loan_decision']} (confidence: {result['probability_bad_loan']:.3f})")
        
        return LoanResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    logger.info("Health check requested")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting Loan Approval API server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")