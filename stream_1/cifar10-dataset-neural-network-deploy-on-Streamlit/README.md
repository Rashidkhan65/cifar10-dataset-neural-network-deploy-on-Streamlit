#CIFAR-10 Dataset Neural Network Deployment on Streamlit
This repository contains code for deploying a convolutional neural network (CNN) trained on the CIFAR-10 dataset using Streamlit. Streamlit is an open-source Python library that makes it easy to create web applications for machine learning and data science projects.

Features
Utilizes the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
Implements a CNN model trained on the CIFAR-10 dataset for image classification.
Deploys the trained model using Streamlit to create an interactive web application.
Allows users to upload their own images and get predictions from the trained model in real-time.
Usage
Clone this repository:
bash
Copy code
git clone https://github.com/your-username/cifar10-dataset-neural-network-deploy-on-Streamlit.git
cd cifar10-dataset-neural-network-deploy-on-Streamlit
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Run the Streamlit application:
bash
Copy code
streamlit run app.py
Once the application is running, open your web browser and go to http://localhost:8501 to access the web interface.
Dependencies
Python 3.x
TensorFlow
Streamlit
NumPy
Matplotlib
File Structure
app.py: Contains the Streamlit application code.
model.py: Defines the CNN model architecture and training process.
utils.py: Utility functions for data preprocessing and visualization.
requirements.txt: List of Python dependencies required to run the application
