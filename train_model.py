import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

import utils

# Path for exported data, numpy arrays
EXPORT_PATH = utils.EXPORT_PATH

# Map actions to numerical labels
label_dict = {label: num for num, label in enumerate(utils.ACTIONS)}

# Initialize lists to store sequences and labels
sequences_list, labels_list = [], []

# Loop through each action and sequence to extract frames
for action in utils.ACTIONS:
    for sequence in range(utils.NUM_SEQUENCES):
        frame_window = []
        for frame_num in range(utils.SEQUENCE_LENGTH):
            try:
                # Load keypoints from numpy files
                keypoints = np.load(os.path.join(EXPORT_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                frame_window.append(keypoints)
            except FileNotFoundError:
                print(f"File not found for action '{action}', sequence '{sequence}', frame '{frame_num}'")
                continue
        sequences_list.append(frame_window)
        labels_list.append(label_dict[action])

# Convert sequences and labels to numpy arrays
X_data = np.array(sequences_list)
Y_data = to_categorical(labels_list).astype(int)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.1)

# Define the directory to save the model
model_directory = os.path.join(os.path.dirname(__file__), 'Saved_Model')
os.makedirs(model_directory, exist_ok=True)

# Define the path for the model file
model_path = os.path.join(model_directory, 'gesture_recognition_model.keras')

# LSTM Model Architecture 
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(utils.ACTIONS), activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=70, validation_split=0.1)

# Save the model
model.save(model_path)

# Display model summary
model.summary()