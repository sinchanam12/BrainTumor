from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load your model (make sure to have tensorflow installed)
from tensorflow.keras.models import load_model

model = load_model(r"C:\Users\Ruchita M Nayak\OneDrive\Attachments\Desktop\Braintumorwebapp\model\brain_tumor_cnn_model.h5")

IMG_SIZE = 100

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_tumor(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    prediction = model.predict(img)[0][0]
    return "Tumor" if prediction > 0.5 else "No Tumor", float(prediction)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get prediction
            label, confidence = predict_tumor(filepath)
            
            return render_template('result.html', 
                                 image=filepath, 
                                 prediction=label,
                                 confidence=f"{confidence:.2f}")
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)