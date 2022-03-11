from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def hello():
    return {"message":"Hello From Financial Advisor!!"}


@app.get("/test")
def hello():
    return {"message":"Testing another endpoint"}

@app.get("/test/{params}")
def hello(params):
    return {"message":"Hello From Financial Advisor!!" + params}