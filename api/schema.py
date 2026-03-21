from pydantic import BaseModel

class LoanApplication(BaseModel):

    loan_amnt: float
    int_rate: float
    annual_inc: float
    dti: float
    installment: float
    revol_util: float
    total_acc: float

    grade: str
    home_ownership: str
    verification_status: str
    purpose: str
    term: str
