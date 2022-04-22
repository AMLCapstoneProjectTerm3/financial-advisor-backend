from datetime import datetime
# import os
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from controller.prediction import generateRandomInputForModel,predictModel, getStocksBasedOnRisk
from models.schemas import BaseResponse, User, RegisterRequest, LoginRequest, PredictRequest, StocksOnRiskLevelRequest
# import numpy as np
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


@app.post("/stocksOnRisk")
def stocksOnRiskLevel(req: StocksOnRiskLevelRequest):
    res = BaseResponse()

    # input_data = {
    #     "investmentAmount": 1000,
    #     "riskLevel": 1
    # }

    input_data = {
        "investmentAmount": int(req.investmentAmount),
        "riskLevel": int(req.riskLevel)
    }

    stocksList = getStocksBasedOnRisk(input_data)
    return {"stocksList": stocksList}


@app.post("/predict")
def predict(req: PredictRequest):
    res = BaseResponse()
    
    print("---------------Predict request received---------------")
    print("risk level: ", req.riskLevel)
    print("user selected stock: ", req.stock)
    print("user's investment amount: ", req.stockAmount)
    print("days to be predicted: ", req.daysToBePredicted)
    # arr = []
    # arr.append(req.stock)
    input_data = {
        "investmentMoney": int(req.stockAmount),
        "riskLevel": int(req.riskLevel),
        "userSelectedStock": req.stock,
        "daysOfPrediction": req.daysToBePredicted
    }
    
    temp = predictModel(input_data)
    
    response = {
        "original_data": temp["original_data"],
        "previous_days_data": temp["previous_days_data"],
        "predicted_days_data": temp["predicted_days_data"]
    }
    res.Success = True
    res.Data = response
    return res

        
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