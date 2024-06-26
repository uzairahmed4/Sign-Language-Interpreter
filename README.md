# REAL-TIME SIGN LANGUAGE INTERPRETER

OVERVIEW

The Sign Language Interpreter project aims to provide real-time translation of sign language gestures into text and speech, 
with the purpose of bridging the communication gap for individuals who use sign language. 
It utilizes MediaPipe Holistic for hand movement tracking and incorporates LSTM architecture to develop a system that interprets the hand gestures.

PROJECT STRUCTURE

utils.py: Contains utility functions used throughout the project.
record_data.py: Script for recording sign language gestures and saving the data in the MP_KeyPoints_Data directory.
train_model.py: Script for training the machine learning model using the recorded data from the MP_KeyPoints_Data directory and saving the trained model in the Saved_Model directory.
real_time_sign_detection.py: Script for running the sign language interpreter in real-time using the trained model.

INSTALLATION

You have two options for installing the required dependencies:

1. Using requirements.txt
Navigate to the project directory in your terminal.
Run the following command to install all dependencies listed in the requirements.txt file:

- pip install -r requirements.txt

2. Installing Individual Packages
Alternatively, you can install each package individually using the following commands:

- pip install opencv-python numpy mediapipe scikit-learn tensorflow pyttsx3
This command will install all necessary dependencies, including OpenCV, NumPy, MediaPipe, scikit-learn, TensorFlow, and pyttsx3.

USAGE 

To run the sign language interpreter, execute the following command:

- python real_time_sign_detection.py
This script uses the trained model stored in the Saved_Model directory to perform real-time sign language translation.

FUTURE DEVELOPMENT

Add more signs into the dataset and increase the instances of training data by collecting more data, and train the model to recognize ASL letters as well.
Create an interface to confirm the sign predictions made by the system before proceeding to a new sentence.
Implementing functionality for navigating back and forth in sign predictions.

## License
This project is licensed under the [MIT License](./LICENSE.txt).

## Contact
For any questions, feedback, or collaborations, feel free to reach out:
- Email: uzairahmedrak@gmail.com
