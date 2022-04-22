from datetime import datetime
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    data: list = Field(None)
    
class LoginRequest(BaseModel):
    password: str  = Field(None)
    email: str = Field(None)
    
class RegisterRequest(BaseModel):
    password: str  = Field(None)
    firstname: str = Field(None)
    lastname: str = Field(None)
    email: str = Field(None)

    
class User(BaseModel):
    firstname: str = Field(None)
    lastname: str = Field(None)
    email: str = Field(None)
    password_hash: str = Field(None)
    date_created: datetime = Field(None)
    
class BaseResponse(BaseModel):
    Success: bool = Field(False)
    ResponseCode: int = Field(None)
    Data: dict = Field({})
    ErrorMessage: str = Field(None)
    
class PredictRequest(BaseModel):
    riskLevel: int = Field(None)
    stock: str = Field(None)
    stockAmount: int = Field(None)
    daysToBePredicted: int = Field(None)

class StocksOnRiskLevelRequest(BaseModel):
    investmentAmount: int = Field(None)
    riskLevel: int = Field(None)