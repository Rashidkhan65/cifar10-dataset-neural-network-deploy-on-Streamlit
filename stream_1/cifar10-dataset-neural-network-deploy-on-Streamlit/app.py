# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.datasets import cifar10
# import matplotlib.pyplot as plt

# # Function to download CIFAR-10 dataset
# def download_cifar10():
#     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#     return (x_train, y_train), (x_test, y_test)

# # Function to display sample images from CIFAR-10 dataset
# def display_samples(x_train, y_train, num_samples=5):
#     st.header("Sample Images from CIFAR-10 Dataset")
#     fig, axes = plt.subplots(1, num_samples, figsize=(10, 10))
#     for i in range(num_samples):
#         axes[i].imshow(x_train[i])
#         axes[i].set_title(f"Label: {y_train[i][0]}")
#         axes[i].axis("off")
#     st.pyplot(fig)

# # Main function to run the Streamlit app
# def main():
#     st.title("CIFAR-10 Dataset Viewer")

#     # Download CIFAR-10 dataset
#     (x_train, y_train), (x_test, y_test) = download_cifar10()

#     # Display sample images
#     display_samples(x_train, y_train)

# # Run the Streamlit app
# if __name__ == "__main__":
#     main()


import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Function to download CIFAR-10 dataset
def download_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)

# Function to display sample images from CIFAR-10 dataset
def display_samples(x_data, y_data, num_samples=5):
    st.header("Testing Images")
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 12))
    for i in range(num_samples):
        axes[i].imshow(x_data[i])
        axes[i].set_title(f"Label: {y_data[i][0]}")
        axes[i].axis("off")
    st.pyplot(fig)

# Function to load pre-trained MobileNetV2 model
def load_model(input_shape=(32, 32, 3), num_classes=10):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Function to evaluate model on CIFAR-10 dataset
def evaluate_model(model, x_test, y_test):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy

# Main function to run the Streamlit app
def main():
    st.set_page_config(layout="wide")

    st.title("CIFAR-10 Dataset Viewer and Model Evaluation")

    # Add sidebar for user input
    st.sidebar.title("Options")
    num_samples = st.sidebar.slider("Number of Samples", min_value=1, max_value=10, value=5)
    dataset_type = st.sidebar.radio("Select Dataset", ("Training", "Testing"))

    # Download CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = download_cifar10()

    # Determine dataset based on user input
    if dataset_type == "Training":
        x_data, y_data = x_train, y_train
    else:
        x_data, y_data = x_test, y_test

    # Display sample images
    display_samples(x_data, y_data, num_samples)

    # Load pre-trained MobileNetV2 model
    model = load_model()

    # Evaluate model on CIFAR-10 dataset
    accuracy = evaluate_model(model, x_test, y_test)
    st.sidebar.write(f"Model Accuracy: {accuracy}")

# Run the Streamlit app
if __name__ == "__main__":
    main()