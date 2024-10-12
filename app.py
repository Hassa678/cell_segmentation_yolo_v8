import subprocess
import shutil
import sys
import os
import logging
from cellSegmentation_v8.pipeline.training_pipeline import TrainPipeline
from cellSegmentation_v8.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from cellSegmentation_v8.constant.application import APP_HOST, APP_PORT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

@app.route("/train")
def trainRoute():
    try:
        obj = TrainPipeline()
        obj.run_pipeline()
        return "Training Successful!!"
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return f"Training failed: {str(e)}", 500

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        yolo_command = [
            "yolo", "task=segment", "mode=predict",
            "model=artifacts/model_trainer/best.pt",
            "conf=0.1", f"source=data/{clApp.filename}", "save=true"
        ]
        
        result = subprocess.run(yolo_command, check=True, capture_output=True, text=True)
        logger.info(f"YOLO output: {result.stdout}")

        output_image_path = os.path.join("runs", "segment", "predict", clApp.filename)
        opencodedbase64 = encodeImageIntoBase64(output_image_path)
        result = {"image": opencodedbase64.decode('utf-8')}

        if os.path.exists("runs"):
            shutil.rmtree("runs")

        return jsonify(result)

    except subprocess.CalledProcessError as e:
        logger.error(f"YOLO execution failed: {e.output}")
        return f"YOLO execution failed: {e.output}", 500
    except ValueError as val:
        logger.error(f"Value error: {str(val)}")
        return Response("Value not found inside JSON data")
    except KeyError as ke:
        logger.error(f"Key error: {str(ke)}")
        return Response("Key value error: incorrect key passed")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return "Invalid input", 500

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT)
    app.run(host='0.0.0.0', port=8080) #for AWS