from fastapi import FastAPI
from firebase import db
app = FastAPI()
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