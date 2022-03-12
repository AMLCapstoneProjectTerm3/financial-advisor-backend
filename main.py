from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

@app.get("/")
def hello():
    return {"message":"Hello From Financial Advisor!!"}

#testing database connection to test collection at firestore
@app.get("/tests")
def hello():
    docs = testdb.stream()
    docs = [x.to_dict() for x in docs]
    return docs

@app.get("/test/{params}")
def hello(params):
    return {"message":"Hello From Financial Advisor!!" + params}

@app.get("/new")
def hello():
    return {"message":"Newwwww"}\
        
@app.get("/predict")
def hello():
    return {"message":"Newwwww"}