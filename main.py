from datetime import datetime
import os
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from controller.prediction import generateRandomInputForModel,predictModel
from models.schemas import BaseResponse, User, RegisterRequest, LoginRequest, PredictRequest
import numpy as np
from firebase import db
from controller.auth import AuthHandler

origins = ["*"]
app = FastAPI()
auth_handler = AuthHandler()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

users = []

testdb = db.collection("test")
usersdb = db.collection("users")
class AuthDetails(BaseModel):
    username: str
    password: str
    
@app.get("/")
def hello():
    return 'Welcome to Financial Advisor API'

#testing database connection to test collection at firestore
@app.get("/tests")
def tests():
    docs = testdb.stream()
    docs = [x.to_dict() for x in docs]
    return docs

@app.post("/predict")
def predict(req: PredictRequest):
    res = BaseResponse()
    
    print("request received", req.riskLevel)
    print("request received", req.stock)
    print("request received", req.stockAmount)
    arr = []
    arr.append(req.stock)
    input_data = {
        "investmentMoney": int(req.stockAmount),
        "riskLevel": int(req.riskLevel),
        "userSelectedStock": arr,
        "daysOfPrediction": 60
    }
    
    print("input created", input_data)
    temp = predictModel(input_data)
    
    # print("temp received", temp)
    print("temp received", temp)
    # print("temp received", temp["previous_days"].tolist())
    # print("temp received", type(temp["previous_days"]))
    # print("temp received", type(temp["previous_days"].tolist()))
    response = {
        "previous_days_data": temp["previous_days_data"],
        "predicted_days_data": temp["predicted_days_data"]
    }
    res.Success = True
    res.Data = response
    return res
    # return {"data": str(predictModel(np.array(req.data))), "success": "True" }
        
@app.get("/predictRandom")
def predictRandom():
    inp = generateRandomInputForModel()
    print(type(inp.tolist()))
    return {"Success": True, "prediction": str(predictModel(inp)), "input_data": inp.tolist()}


@app.post('/register', status_code=201)
def register(req: RegisterRequest):
    
    res = BaseResponse()
    try:
        # checking if user already exists for an email
        docs = usersdb.where('email', '==', str(req.email)).stream()
        docs = [x.to_dict() for x in docs]
        if(len(docs) > 0):
            res.ResponseCode = 401
            res.ErrorMessage = "User already exists"
            res.Data = {}
            return res

        #hashing the password to save in db
        hashed_password = auth_handler.get_password_hash(req.password)
        
        # create user
        user = User()
        user.firstname = req.firstname
        user.lastname = req.lastname
        user.email = req.email
        user.password_hash = hashed_password
        user.date_created = datetime.now()
      
        # save user to db
        usersdb.add(dict(user))

        res.Data = dict(user)
        res.Success = True
        res.ResponseCode = 201
        res.ErrorMessage = ''
        return res
    except:
        res.ResponseCode = 500
        res.ErrorMessage = "An Exception has occurred"
        res.Data = {}
        return res


@app.post('/login')
def login(req: LoginRequest):
    res = BaseResponse()
    try:
        # getting the user form db
        docsRef = usersdb.where('email', '==', str(req.email)).limit(1).stream()
        docs = []
        for doc in docsRef:
            docid = doc.id
            data = doc.to_dict()
            data.update({'docid': docid})
            docs.append(data)
        
        # docs = [x.to_dict() and x.update({'docid': x.id}) for x in docs]
        print(docs[0])
        
        # verify integrity
        if(len(docs) == 0 or (not auth_handler.verify_password(req.password, docs[0]['password_hash']))):
            res.ErrorMessage = 'Username or password does not match'
            res.ResponseCode = 402
            res.Data = {}
            return res
        
        print(docs[0])
        
        # generate token
        token = auth_handler.encode_token(docs[0]['docid'])
        
        res.Success = True
        res.ResponseCode = 201
        res.Data = {
            'token': 'Bearer ' + str(token)
        }
        return res
    except:
        res.ErrorMessage = 'Server responded with error. Please try again after some time'
        res.ResponseCode = 500
        res.Data = {}
        return res

@app.get('/protected')
def protected(username=Depends(auth_handler.auth_wrapper)):
    return { 'name': username }

@app.get("/userdetails")
async def userdetails(username=Depends(auth_handler.auth_wrapper)):
    res = BaseResponse()
    if(username):    
        print("Getting user details for username: " + str(username))
        doc_ref = usersdb.document(username)
        doc = doc_ref.get()
        print(doc)
        print(doc.exists)
        if doc.exists:
            print("doc exists")
            doc = doc.to_dict();
            print(doc)
            res.Success = True;
            res.Data = {
                "email": doc['email'],
                "displayName": doc['firstname'] + " " + doc['lastname'],
                "date_created": doc['date_created']
            }
            
            print(res)
            return res;
        else:
            res.Success = True
            res.ErrorMessage = "No user details found"
            res.Data = {}
            
            return res;
    else:
        res.Success = False
        res.ErrorMessage = "User not Found please try again"
        res.Data = {}
        
        return res;