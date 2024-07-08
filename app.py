from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt
from ultralytics import YOLO


# Initialize Flask application
app = Flask(__name__)

# Define the upload folder location
UPLOAD_FOLDER = 'clustering/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained KMeans model
kmeans_model = joblib.load('clustering/kmeans_model.pkl')

# Load the saved encoders and scaler
label_encoder_code_module = joblib.load('clustering/encoder/label_encoder_code_module.pkl')
label_encoder_highest_education = joblib.load('clustering/encoder/label_encoder_highest_education.pkl')
label_encoder_imd_band = joblib.load('clustering/encoder/label_encoder_imd_band.pkl')
label_encoder_age_band = joblib.load('clustering/encoder/label_encoder_age_band.pkl')
scaler = joblib.load('clustering/scaler/scaler.pkl')


# Load your machine learning models and necessary variables
# Example: Load your LR model and class_names
lr_model = joblib.load("classification/logistic_regression.pkl")  # Load your Logistic Regression model
model_yolo = YOLO('classification/yolov8m.pt') 
class_names = ['distracted', 'fatigue', 'focused', 'raise_hand', 'sleeping', 'using_smartphone', 'writing_reading']

# Constants for image processing
img_size = (128, 128)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Function to preprocess and predict image
def predict_image(image_path, model, class_names):
    """Predicts the class of a single image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Placeholder for Object Detection with YOLOv8 (replace with actual logic)
    # Assuming YOLOv8 is used for object detection and cropping
    results = model_yolo(img) 
    
    # Placeholder for Object Detection results
    x1, y1, x2, y2 = 0, 0, img.shape[1], img.shape[0]  # Default to full image size

    # Square Cropping
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    crop_size = max(x2 - x1, y2 - y1)  # Size of the larger side

    crop_x1 = max(0, center_x - crop_size // 2)
    crop_y1 = max(0, center_y - crop_size // 2)
    crop_x2 = min(img.shape[1], crop_x1 + crop_size)
    crop_y2 = min(img.shape[0], crop_y1 + crop_size)

    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize cropped image to desired size
    cropped_img = cv2.resize(cropped_img, img_size)
    # Preprocess for VGG16
    cropped_img = image.img_to_array(cropped_img)
    cropped_img = preprocess_input(cropped_img)
    cropped_img = np.expand_dims(cropped_img, axis=0)

    # Extract features using VGG16 base model
    features = base_model.predict(cropped_img)
    features = features.reshape(features.shape[0], -1)

    # Predict using the loaded LR model
    prediction = model.predict(features)[0]
    predicted_class = class_names[prediction]

    return predicted_class

# Function to process video and extract frames
def extract_frames(video_path, num_frames=50):
    """Extracts frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, frame_count - 1, num=num_frames, dtype=np.int32)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames

@app.route('/')
def home():
    return render_template('classification.html')

# Route for classification page (handles both image and video uploads)
@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == 'POST':
        # Handle file upload and prediction here
        uploaded_files = request.files.getlist('images')
        video_file = request.files['video'] if 'video' in request.files else None

        if video_file:
            # Save the uploaded video
            video_filename = video_file.filename
            video_path = os.path.join('classification/uploads', video_filename)
            video_file.save(video_path)

            # Extract frames from the video
            frames = extract_frames(video_path)

            # Process each frame and predict
            results = []
            for frame in frames:
                # Temporarily save frame as an image
                temp_image_path = 'classification/uploads/temp_frame.jpg'
                cv2.imwrite(temp_image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                # Predict the class of the frame
                predicted_class = predict_image(temp_image_path, lr_model, class_names)
                results.append(predicted_class)

                # Remove temporary image
                os.remove(temp_image_path)

            # Generate label distribution plot
            labels = class_names
            values = [results.count(cls) for cls in class_names]

            plt.figure(figsize=(12, 6))
            plt.bar(labels, values, color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink'])
            plt.xlabel('Engagement Level')
            plt.ylabel('Count')
            plt.title('Student Engagement Level Distribution')

            # Save the plot to a static file
            plot_path = 'static/uploads/label_distribution.png'
            plt.savefig(plot_path)
            plt.close()

            # Render the classification template with the plot
            return render_template('classification.html', plot_path=plot_path)

        elif uploaded_files:
            # Process each uploaded image
            results = []
            for file in uploaded_files:
                # Save the uploaded file
                filename = file.filename
                file_path = os.path.join('classification/uploads', filename)
                file.save(file_path)

                # Predict the class of the uploaded image
                predicted_class = predict_image(file_path, lr_model, class_names)
                results.append(predicted_class)

            # Generate label distribution plot
            labels = class_names
            values = [results.count(cls) for cls in class_names]

            plt.figure(figsize=(8, 6))
            plt.bar(labels, values, color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink'])
            plt.xlabel('Engagement Level')
            plt.ylabel('Count')
            plt.title('Label Distribution')

            # Save the plot to a static file
            plot_path = 'static/uploads/label_distribution.png'
            plt.savefig(plot_path)
            plt.close()

            # Render the classification template with the plot
            return render_template('classification.html', plot_path=plot_path)

    # Render the classification template initially
    return render_template('classification.html')


@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    if request.method == 'POST':
        if 'single_record' in request.form:
            id_student = request.form['id_student']
            code_module = request.form['code_module']
            sum_click = int(request.form['sum_click'])  # Convert to int for compatibility
            gender = request.form['gender']
            highest_education = request.form['highest_education']
            imd_band = request.form['imd_band']
            age_band = request.form['age_band']
            studied_credits = int(request.form['studied_credits'])  # Convert to int for compatibility
            disability = request.form['disability']

            # Store the original input data for display
            input_data = {
                'id_student': id_student,
                'code_module': code_module,
                'sum_click': sum_click,
                'gender': gender,
                'highest_education': highest_education,
                'imd_band': imd_band,
                'age_band': age_band,
                'studied_credits': studied_credits,
                'disability': disability
            }

            # Preprocess the input for prediction
            # Convert gender and disability to binary
            gender_bin = 1 if gender == 'M' else 0
            disability_bin = 1 if disability == 'Y' else 0

            # Encode categorical variables
            code_module_encoded = label_encoder_code_module.transform([code_module])[0]
            highest_education_encoded = label_encoder_highest_education.transform([highest_education])[0]
            imd_band_encoded = label_encoder_imd_band.transform([imd_band])[0]
            age_band_encoded = label_encoder_age_band.transform([age_band])[0]

            # Scale numerical variables
            sum_click_scaled, studied_credits_scaled = scaler.transform([[sum_click, studied_credits]])[0]

            # Predict the cluster
            cluster = kmeans_model.predict([[code_module_encoded, sum_click_scaled, gender_bin,
                                             highest_education_encoded, imd_band_encoded,
                                             age_band_encoded, studied_credits_scaled, disability_bin]])[0]

            # Add the cluster to the input data
            input_data['cluster'] = cluster

            return render_template('clustering.html', single_record=True, input_data=input_data)

        elif 'batch_upload' in request.files:
            file = request.files['batch_upload']
            if file:
                # Save the uploaded file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                # Read CSV and make a copy for processing
                original_df = pd.read_csv(file_path)
                processed_df = original_df.copy()

                # Preprocess the batch input for prediction
                processed_df['gender'] = processed_df['gender'].apply(lambda x: 1 if x == 'M' else 0)
                processed_df['disability'] = processed_df['disability'].apply(lambda x: 1 if x == 'Y' else 0)
                processed_df['code_module'] = label_encoder_code_module.transform(processed_df['code_module'])
                processed_df['highest_education'] = label_encoder_highest_education.transform(processed_df['highest_education'])
                processed_df['imd_band'] = label_encoder_imd_band.transform(processed_df['imd_band'])
                processed_df['age_band'] = label_encoder_age_band.transform(processed_df['age_band'])
                processed_df['sum_click'] = processed_df['sum_click'].astype(int)
                processed_df['studied_credits'] = processed_df['studied_credits'].astype(int)
                processed_df[['sum_click', 'studied_credits']] = scaler.transform(processed_df[['sum_click', 'studied_credits']])

                # Predict clusters
                processed_df['cluster'] = kmeans_model.predict(processed_df.drop(columns=['id_student']))

                # Add predicted clusters to the original DataFrame
                original_df['cluster'] = processed_df['cluster']

                # Calculate percentages of students in each cluster
                cluster_counts = original_df['cluster'].value_counts(normalize=True).sort_index() * 100
                clusters = cluster_counts.index.tolist()
                percentages = cluster_counts.values.tolist()

                plt.figure(figsize=(12, 6))
                bars = plt.barh(clusters, percentages, color=['orange', 'blue'])  # Customize colors here

                plt.xlabel('Percentage of Students')
                plt.ylabel('Cluster')
                plt.title('Cluster Distribution')
                plt.yticks(clusters, ['Cluster 0', 'Cluster 1'])  # Custom tick labels

                # Adding percentage labels to the bars
                for bar, percentage in zip(bars, percentages):
                    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{percentage:.2f}%', va='center')

                plot_path = './static/uploads/clustered_plot.png'
                plt.savefig(plot_path)
                plt.close()

                # Return rendered template with results
                return render_template('clustering.html', table=original_df.to_html(classes='table table-striped', index=False), plot=True)

    return render_template('clustering.html')


# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
