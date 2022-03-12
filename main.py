from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from controller.prediction import generateRandomInputForModel,predictModel
import numpy as np
origins = ["*"]
from firebase import db
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

testdb = db.collection("test")

class PredictRequest(BaseModel):
    data: list = None
    
@app.get("/")
def hello():
    return {"message":"Hello From Financial Advisor!!"}

#testing database connection to test collection at firestore
@app.get("/tests")
def tests():
    docs = testdb.stream()
    docs = [x.to_dict() for x in docs]
    return docs

@app.post("/predict")
def predict(req: PredictRequest):
    return {"data": str(predictModel(np.array(req.data))), "success": "True" }
        
@app.get("/predictRandom")
def predictRandom():
    inp = generateRandomInputForModel()
    print(type(inp.tolist()))
    return {"Success": True, "prediction": str(predictModel(inp)), "input_data": inp.tolist()}

