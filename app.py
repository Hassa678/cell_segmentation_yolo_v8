import sys,os
from cellSegmentation_v8.pipeline.training_pipeline import TrainPipeline
from cellSegmentation_v8.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin


obj = TrainPipeline()
obj.run_pipeline()
print ("training done")