import logging
import uuid
import os
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import shutil
from typing import List
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import mimetypes

from detect_face import TransformImageToPassportSpecs

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

transformer = TransformImageToPassportSpecs()


# %% ---------------------------------------------
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    new_name = str(uuid.uuid4()).split("-")[0]
    ext = file.filename.split(".")[-1]
    file_path = f"static/uploads/{new_name}.{ext}"
    print(file_path)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return templates.TemplateResponse(
        "index.html", {"request": request, "filename": f"{new_name}.{ext}"}
    )


@app.get("/process/{filename}")
def process_file(request: Request, filename: str):
    # Processing logic goes here
    image_path = (
        f"static/uploads/{filename}"  # Replace with actual path of processed image
    )

    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    output_image = transformer.transform(input_image)

    output_path = os.path.join("static/downloads", filename)
    output_image.save(output_path)

    return templates.TemplateResponse(
        "index.html", {"request": request, "processed_image": filename}
    )


@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = f"static/downloads/{filename}"
    media_type, _ = mimetypes.guess_type(file_path)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return FileResponse(
        file_path,
        media_type=media_type,
        headers=headers,
    )
