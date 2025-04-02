import uvicorn
from transformers import pipeline 
from fastapi import FastAPI
from pydantic import BaseModel
 

app = FastAPI()

 
class InputData(BaseModel):
    feature: float

class PredictionResult(BaseModel):
    prediction: float


@app.post('/predict', response_model=PredictionResult)
def predict(input_data: InputData):

    # input_array = np.array([[input_data.feature]])
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    # prediction = model.predict(input_array)

    return {"prediction": 32.5}

@app.get('/')
async def root(): 
    return {"message": "Panin to the FastAPI model prediction service!"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)