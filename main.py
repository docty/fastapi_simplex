import uvicorn
import numpy as np
from fastapi import FastAPI

app = FastAPI()


@app.get('/')
async def root():
    a = np.array([42])
    print(a[0])
    return {"message": a.tolist()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)