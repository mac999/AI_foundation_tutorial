
import os, json
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, Query
from typing import List, Dict, Any, Union
from PIL import Image
from io import BytesIO

app = FastAPI()

# Load the YOLOv8 model
model_path = os.path.join(os.path.dirname(__file__), "yolov8m.pt")
model = YOLO(model_path)

def inference_on_path(imgs_path: str) -> List[Dict[str, Any]]:
	"""
	Runs inference on the YOLOv8 architecture for the images available in the given path
	"""
	results: ultralytics.engine.results.Results = model(source=imgs_path, show=False, conf=0.45)
	results_data = []
	for result in results:
		result_metadata = {
			'shape': result.orig_shape,
			'path': result.path,
			'detections': json.loads(result.tojson())
		}
		results_data.append(result_metadata)

	return results_data

def inference_on_img(img: Image) -> List[Dict[str, Any]]:
	"""
	Runs inference on the YOLOv8 architecture for the given image
	"""
	results: ultralytics.engine.results.Results = model(source=img, show=False, conf=0.45)
	result_data = json.loads(results[0].tojson())

	return result_data

@app.get('/')
def home() -> Dict[str, Union[int, Dict[str, Any]]]:
    """Ping method for checking API status"""
    return {
        'status_code': 200,
        'data': {
            'api_health_status': 'OK'
        }}


@app.get('/detect')
def detect(
    path: str = Query(..., description='Path to the images directory')
) -> Dict[str, Union[int, Dict[str, Any]]]:
    """
    Runs YOLO inference for all the images available in the path received as query parameter
    """
    path = path.replace('$', '/')

    # Run inference for the images on the given path 
    try:
        inference_results_data = inference_on_path(imgs_path=path)
    except Exception as err:
        print(f'An error occurred while trying to perform inference. {err}')
        return {
            'status_code': 500,
            'data': {}
        }

    return {
        'status_code': 200, 
        'data': {'inference_results': inference_results_data}
    }

@app.post('/detect_img')
async def detect_img(
    img: UploadFile = File(...)
) -> Dict[str, Union[int, Dict[str, Any]]]:
    """Runs YOLO inference for the image received in the payload"""
    if not img:
        return {
            'status_code': 400, 
            'data': {'message': 'No upload file sent'}
        }

    # Convert the received image to PIL Image
    img_content = await img.read()
    image_stream = BytesIO(img_content)
    image = Image.open(image_stream)

    # Run inference on the received image
    try:
        inference_results_data = inference_on_img(img=image)
    except Exception as err:
        print(f'An error occurred while trying to perform inference. {err}')
        return {
            'status_code': 500,
            'data': {}
        }

    return {
        'status_code': 200, 
        'data': {
            'image_name': img.filename,
            'image_size': img.size,
            'inference_results': inference_results_data
        }
    }

@app.get('/images')
def get_available_images(
    path: str = Query(..., description='The path to search for images')
) -> Dict[str, Union[int, Dict[str, Any]]]:
    """Returns the list of available PNG and JPG image files in the path received as query parameter"""
    available_images = []
    path = path.replace('$', '/')

    # Retrieve all images available for inference (JPG and PNG)
    if os.path.exists(path):
        available_images = [filename for filename in os.listdir(path) if filename.endswith(('.png', '.jpg'))]
        return {
            'status_code': 200,
            'data': {
                'available_images': available_images, 
                'size': len(available_images)
            }
        }

    return {
        'status_code': 400, 
        'data': {
            'message': 'The provided path does not exist', 
            'size': 0
        }
    }
