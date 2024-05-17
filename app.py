from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    if 'frame' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    num_of_faces = len(faces)
    return jsonify({"num_of_faces": num_of_faces})

if __name__ == '__main__':
    app.run(debug=True)
