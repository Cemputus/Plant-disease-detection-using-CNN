import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Create the uploads directory if it doesn't exist
upload_folder = os.path.join('static', 'uploads')
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files.get('image')
        if not image or not hasattr(image, 'filename') or image.filename is None or image.filename.strip() == '':
            return render_template('submit.html', error="No file selected. Please upload a clear image of a plant leaf.")
        # Generate a unique filename like camera_image1.jpg, camera_image2.jpg, ...
        base_name = "camera_image"
        ext = os.path.splitext(secure_filename(image.filename))[1] or ".jpg"
        existing = [f for f in os.listdir(upload_folder) if f.startswith(base_name) and f.endswith(ext)]
        nums = [int(f[len(base_name):-len(ext)]) for f in existing if f[len(base_name):-len(ext)].isdigit()]
        next_num = max(nums) + 1 if nums else 1
        unique_filename = f"{base_name}{next_num}{ext}"
        file_path = os.path.join(upload_folder, unique_filename)
        image.save(file_path)
        pred = prediction(file_path)
        valid_leaf_indices = [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
        if pred not in valid_leaf_indices:
            suggestions = [
                "Ensure the image is a clear, close-up photo of a healthy or diseased plant leaf.",
                "Avoid uploading images of backgrounds, soil, or non-leaf objects.",
                "For best results, use natural lighting and avoid shadows on the leaf.",
                "If you need help with farming or plant disease control, consider these tips:",
                "- Practice crop rotation and use disease-resistant varieties.",
                "- Remove and destroy infected plants promptly.",
                "- Maintain proper spacing for air circulation.",
                "- Use organic or recommended chemical treatments as needed.",
                "- Keep your tools and hands clean to prevent disease spread."
                "- Emmanuel Nsubuga @cenAnalytics  "
            ]
            rel_path = os.path.relpath(file_path, 'static').replace('\\', '/')
            return render_template('submit.html', error="Please upload a clear image of a plant leaf for accurate disease detection.", suggestions=suggestions, uploaded_image_path=rel_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        rel_path = os.path.relpath(file_path, 'static').replace('\\', '/')
        return render_template('submit.html', title=title, desc=description, prevent=prevent, 
                               image_url=image_url, pred=pred, sname=supplement_name, simage=supplement_image_url, buy_link=supplement_buy_link, uploaded_image_path=rel_path)
    return render_template('submit.html')

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
