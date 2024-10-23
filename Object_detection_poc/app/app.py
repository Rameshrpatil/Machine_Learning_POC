from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import io
import os

app = Flask(__name__)

#the standard YOLO model  
standard_model = YOLO('./model/yolov8s.pt')

#custom YOLO model  
custom_model = YOLO('./model/best.pt') 

 
result_dir = 'result'
os.makedirs(result_dir, exist_ok=True)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(result_dir, filename)
    image = Image.open(image_file.stream).convert("RGB")

 
    standard_results = standard_model(image)
    custom_results = custom_model(image)

 
    standard_class_ids = standard_results[0].boxes.cls
    custom_class_ids = custom_results[0].boxes.cls

 
    standard_detected_classes = [standard_model.names[int(cls_id)] for cls_id in set(standard_class_ids)]
    custom_detected_classes = [custom_model.names[int(cls_id)] for cls_id in set(custom_class_ids)]

 
    detected_classes = list(set(standard_detected_classes + custom_detected_classes))

 
    custom_results[0].plot()  
    custom_results[0].save(image_path) 

    return jsonify({
        "detected_classes": detected_classes,
        "image_location": image_path
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)