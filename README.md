# Plant Disease Detection using Deep Learning 🌿

A beginner-friendly CNN project for classifying plant leaf images using TensorFlow/Keras.

## Project structure
- `train.py` - trains the CNN model from a dataset
- `predict.py` - predicts the disease from a single leaf image
- `requirements.txt` - required Python packages
- `dataset/README.txt` - dataset folder instructions
- `models/` - trained model is saved here
- `predictions/` - put images here for prediction

## Dataset
This project is designed for the PlantVillage dataset.

Download a PlantVillage image dataset and arrange it like this:

dataset/
  Apple___Apple_scab/
    image1.jpg
    image2.jpg
  Apple___Black_rot/
    image1.jpg
  ...

Each disease/healthy class must have its own folder.

## Run in Google Colab or VS Code

1. Install Python 3.10+.
2. Open the project folder in VS Code.
3. Open Terminal.
4. Run:
   pip install -r requirements.txt
5. Put your dataset inside `dataset/`.
6. Train:
   python train.py
7. Put a test leaf image inside `predictions/`.
8. Run:
   python predict.py predictions/leaf.jpg

The trained model will be saved as:
models/plant_disease_model.keras

## Important
The ZIP does not contain the PlantVillage image dataset because the dataset is large. Add the dataset to the `dataset` folder before training.
