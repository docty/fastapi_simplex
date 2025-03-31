import uvicorn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort

app = FastAPI()

sess = ort.InferenceSession('./model.onnx')

class InputData(BaseModel):
    feature: float

class PredictionResult(BaseModel):
    prediction: float


@app.post('/predict', response_model=PredictionResult)
def predict(input_data: InputData):

    input_array = np.array([[input_data.feature]], dtype=np.float32)

    input_name = sess.get_inputs()[0].name
    #output_name = sess.get_outputs()[0].name
    prediction = sess.run(None, {input_name: input_array})[0]

    

    return {"prediction": float(prediction[0][0])}

@app.get('/')
async def root(): 
    return {"message": "Welcome to the FastAPI model prediction service!"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)