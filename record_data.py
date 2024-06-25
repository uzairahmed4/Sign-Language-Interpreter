import cv2
import numpy as np
import os

import utils

# Initialize Mediapipe Holistic model
holistic_model = utils.holistic_model.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Path for exported data, numpy arrays
EXPORT_PATH = utils.EXPORT_PATH

# Actions that we try to detect
ACTIONS = utils.ACTIONS

# Number of sequences (videos) to be recorded
NUM_SEQUENCES = utils.NUM_SEQUENCES

# Length of each sequence (number of frames per video)
SEQUENCE_LENGTH = utils.SEQUENCE_LENGTH

# Method call to create data folders
utils.create_data_folders(ACTIONS, NUM_SEQUENCES, EXPORT_PATH)

# Set up Mediapipe Holistic model
with holistic_model as holistic:
    
    while True:  # Main loop to handle user input

        # Prompt the user to choose an action to record
        print("Choose an action to record (enter the number corresponding to the action):")
        for i, action in enumerate(ACTIONS, start=1):
            print(f"{i}. {action}")
        print("Press 'x' to exit.")

        # Get user input for action choice
        user_input_action = input()

        if user_input_action.lower() == 'x':
            break  # Exit if the user inputs 'x'

        try:
            action_choice = int(user_input_action)
            if action_choice < 1 or action_choice > len(ACTIONS):
                print("Invalid choice. Please enter a number within the range.")
                continue
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            continue

        action = ACTIONS[action_choice - 1]

        # Prompt the user to choose a batch
        print("Choose a batch number (1, 2, or 3):")
        user_input_batch = input()

        try:
            batch_choice = int(user_input_batch)
            if batch_choice not in [1, 2, 3]:
                print("Invalid choice. Please enter 1, 2, or 3.")
                continue
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            continue

        # Determine the range of sequences for the chosen batch
        start_sequence = (batch_choice - 1) * 20
        end_sequence = batch_choice * 20

        # Open camera feed
        cap = cv2.VideoCapture(0)

        # Loop through each sequence (video)
        for sequence in range(start_sequence, end_sequence):

            # Loop through each frame in the sequence
            for frame_num in range(SEQUENCE_LENGTH):

                # Read frame from camera feed
                ret, frame = cap.read()

                # Check if the camera feed is working
                if not ret:
                    break  # Exit the loop if unable to capture frame from the camera

                # Make detections using Mediapipe Holistic model
                image, results = utils.detect_keypoints(frame, holistic_model)

                # Draw landmarks on the frame
                utils.draw_landmarks(image, results)

                # Display message to indicate starting collection
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                # Show frame on screen
                cv2.imshow('Camera', image)

                # Export keypoints to numpy file
                keypoints = utils.extract_keypoints(results)
                npy_path = os.path.join(EXPORT_PATH, action, str(sequence), str(frame_num))
                
                # Remove existing numpy file, if any
                if os.path.exists(npy_path + '.npy'):
                    os.remove(npy_path + '.npy')
                
                np.save(npy_path, keypoints)

                # Check for 'x' key event to exit recording
                if cv2.waitKey(1) & 0xFF == ord('x') or cv2.waitKey(1) & 0xFF == ord('X'):
                    print("Exiting capture...")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()  # Exit the script immediately
                    break  # Exit the inner loop

            # Add delay after capturing all frames for a sequence
            cv2.waitKey(2500)

        # Release the camera after recording the batch
        cap.release()
        cv2.destroyAllWindows()
