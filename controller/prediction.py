from keras.models import load_model
import numpy as np
model = load_model("./models/financial_ad_model.h5")


def generateRandomInputForModel():
    return np.random.random(100).reshape(1,100,1)

def predictModel(inp):
    try:
        return model.predict(inp)[0][0]
    except :
        return "System Error: Could not predict"