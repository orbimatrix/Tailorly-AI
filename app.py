from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from measurement import BodyMeasurement
import uvicorn
import os

app = FastAPI()

# Initialize measurement logic
measurer = BodyMeasurement()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/measure")
async def measure_body(image: UploadFile = File(...), height: float = Form(...)):
    try:
        contents = await image.read()
        results = measurer.process_image(contents, height)
        return results
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
