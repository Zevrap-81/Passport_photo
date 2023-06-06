# Passport Photo Maker App

Welcome to the Passport Photo Maker App! This application allows you to easily create passport photos using AI-powered image processing techniques.

## Features

- Upload an image and process it to meet passport photo specifications.
- Download the processed passport photo.
- Responsive user interface with Bootstrap styling.

## Installation with DevContainers in VS Code

To set up the development environment using DevContainers in VS Code, please follow these steps:

1. Ensure that you have Visual Studio Code installed on your system.
2. Clone the repository to your local machine.
3. Open the repository folder in Visual Studio Code.
4. Install the "Remote - Containers" extension by Microsoft from the VS Code extensions marketplace.
5. Click on the green "><" icon in the lower-left corner of the VS Code window and select "Reopen in Container" from the popup.
6. VS Code will now build and start the development container. This may take a few minutes depending on your system.
7. Once the container is up and running, run the following commands 
```
mkdir model_ckpts 
cd model_ckpts 
gdown --id 1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ
wget -P . https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
```
These commands will download U2net and mediapipe face landmark detector models
## Serving the App with Uvicorn

To serve the app locally using Uvicorn, run 

```
uvicorn app:app --port 8000
```
you can access the app at http://localhost:8000.