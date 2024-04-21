from fastapi import FastAPI,UploadFile,File,Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL=tf.keras.models.load_model('./models/1')
CLASS_NAMES=['Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post('/predict')
async def predict(file:UploadFile=File()):
    img_file=await file.read()
    image=np.array(Image.open(BytesIO(img_file)))
    image_batch=np.expand_dims(image,0)
    predictions=MODEL.predict(image_batch)
    pred_class=CLASS_NAMES[np.argmax(predictions[0])]
    conf=round(100*(np.max(predictions[0])),2)
    print(pred_class,conf)
    return {
        "class":pred_class,
        "confidence":float(conf)
    }

if __name__=="__main__":
    uvicorn.run(app,port=8000,host='localhost')
