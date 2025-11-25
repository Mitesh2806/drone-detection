# 🚁 YOLOv8 Drone & Aerial Object Detection

A real-time object detection web application built with **Streamlit** and **Ultralytics YOLOv8**. This application allows users to upload images to detect and classify aerial objects such as Drones, Airplanes, and Helicopters.

## 🌟 Features

* **Object Detection:** Accurately detects aerial objects in uploaded images using a custom-trained YOLOv8 model.
* **Supported Classes:** Detects multiple classes including **Drones**, **Airplanes**, and **Helicopters**.
* **Interactive Interface:** Built with Streamlit for a responsive and easy-to-use web experience.
* **Adjustable Confidence:** Includes a sidebar slider to filter predictions based on model confidence (default 0.25).
* **Visual Output:** Displays the original image alongside the processed image with bounding boxes around detected objects.
* **Detailed Data:** Provides a data table showing coordinates (`x1`, `y1`, `x2`, `y2`), class names, and confidence scores for every detection.

## 🛠️ Tech Stack

* **Python 3.8+**
* **Streamlit:** Web application framework.
* **YOLOv8 (Ultralytics):** State-of-the-art object detection model.
* **OpenCV & PIL:** Image processing libraries.
* **Pandas:** Data manipulation and analysis.
* **Torch:** Deep learning framework.

## 📂 Project Structure

```text
├── app.py              # Main Streamlit application script
├── best.pt             # Custom trained YOLOv8 model weights
├── requirements.txt    # List of Python dependencies
└── results.csv         # Training metrics and performance logs
```
```Bash
git clone <your-repo-url>
cd <your-repo-directory>
Create a Virtual Environment (Recommended)
```

```Bash

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
Install Dependencies

```

pip install -r requirements.txt
Run the Application Ensure the best.pt file is in the same directory as app.py, then run:

```

streamlit run app.py
The app will open in your browser at http://localhost:8501.

## 📊 Model Performance
The model (best.pt) was trained over 50 epochs. According to the training results:

mAP50 (Mean Average Precision @ 0.5 IoU): ~96.6%

Precision: ~93.0%

Recall: ~94.5%

The model shows strong convergence with consistently decreasing box and classification loss over the training period.

## 🖥️ Usage Guide
Upload Image: Click the "Browse files" button to upload an image (supported formats: JPG, PNG, JPEG).

Adjust Threshold: Use the "Confidence Threshold" slider on the sidebar to fine-tune detection sensitivity.


## 📜 License
This project utilizes Ultralytics YOLOv8, which is licensed under the AGPL-3.0 license. Please ensure you comply with this license for any commercial or open-source use.
