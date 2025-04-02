import uvicorn
from transformers import pipeline 
from fastapi import FastAPI
from pydantic import BaseModel
 

app = FastAPI()
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

candidate_labels = ["education", "health", "business", "sports", "politics", "entertainment"]

 
class InputData(BaseModel):
    text: str

class PredictionResult(BaseModel):
    prediction: str


@app.post('/predict', response_model=PredictionResult)
def predict(input_data: InputData):

    #result = classifier(input_data.text, candidate_labels)
    #scores = result['scores']
    #labels = result['labels'][0]
    #max_score_index = scores.index(max(scores))


    return {"prediction": 'input_data'}

@app.get('/')
async def root(): 
    return {"message": "Everything you to the FastAPI model prediction service!"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)