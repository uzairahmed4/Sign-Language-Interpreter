import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize Mediapipe Holistic model and drawing utilities
holistic_model = mp.solutions.holistic  # Holistic model
drawing_utils = mp.solutions.drawing_utils  # Drawing utilities

def detect_keypoints(image, model):
    """
    Detects keypoints in an image using a specified model (MediaPipe Holistic).

    Args:
        image: The input image.
        model: The model to use for keypoint detection.

    Returns:
        tuple: A tuple containing the processed image with keypoints and the detection results.
            The processed image is in BGR format.
            The detection results contain information about detected keypoints.
    """
    
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Set image flags to non-writable
    image_rgb.flags.writeable = False
    # Make predictions
    results = model.process(image_rgb)
    # Set image flags to writable again
    image_rgb.flags.writeable = True
    # Convert image from RGB back to BGR
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, results

def draw_landmarks(image, results):
    """
    Draws landmarks on an image based on detected keypoints.

    Args:
        image: The input image.
        results: The detection results containing information about detected keypoints.

    Returns:
        None
    """
    
    # Draw pose connections
    drawing_utils.draw_landmarks(image, results.pose_landmarks, holistic_model.POSE_CONNECTIONS,
                                 drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4), 
                                 drawing_utils.DrawingSpec(color=(0, 128, 255), thickness=2, circle_radius=2)
                                 ) 
    # Draw left hand connections
    drawing_utils.draw_landmarks(image, results.left_hand_landmarks, holistic_model.HAND_CONNECTIONS, 
                                 drawing_utils.DrawingSpec(color=(0, 128, 0), thickness=2, circle_radius=4), 
                                 drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
                                 ) 
    # Draw right hand connections  
    drawing_utils.draw_landmarks(image, results.right_hand_landmarks, holistic_model.HAND_CONNECTIONS, 
                                 drawing_utils.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=4), 
                                 drawing_utils.DrawingSpec(color=(255, 128, 0), thickness=2, circle_radius=2)
                                 ) 

def extract_keypoints(results):
    """
    Extracts keypoints from the detection results.

    Args:
        results: The detection results containing information about detected keypoints.

    Returns:
        An array containing the extracted keypoints.
            The array contains concatenated pose, left hand, and right hand keypoints.
            Each set of keypoints is flattened into a single dimension.
            If no keypoints are found, zeros are filled in.
    """
    
    keypoints = []
    
    # Extract pose keypoints
    if results.pose_landmarks:
        pose_keypoints = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten()
    else:
        pose_keypoints = np.zeros(33 * 4)

    keypoints.append(pose_keypoints)

    # Extract left hand keypoints
    if results.left_hand_landmarks:
        left_hand_keypoints = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand_keypoints = np.zeros(21 * 3)

    keypoints.append(left_hand_keypoints)

    # Extract right hand keypoints
    if results.right_hand_landmarks:
        right_hand_keypoints = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand_keypoints = np.zeros(21 * 3)

    keypoints.append(right_hand_keypoints)

    # Concatenate all keypoints into a single array
    return np.concatenate(keypoints)

def create_data_folders(ACTIONS, NUM_SEQUENCES, EXPORT_PATH):
    """
    Creates folder structure for storing exported data.
    Each action has subfolders for each sequence.

    Args:
        ACTIONS: Array of actions to be detected.
        NUM_SEQUENCES: Number of sequences (videos) to be recorded.
        EXPORT_PATH: Path for exporting data.

    Returns:
        None
    """
    # Get the path of the current script
    script_path = os.path.abspath(__file__)
    # Get the parent directory of the script
    sign_folder = os.path.dirname(script_path)
    # Create folders for each action and each sequence within the action
    for action in ACTIONS: 
        for sequence in range(NUM_SEQUENCES):
            try: 
                os.makedirs(os.path.join(sign_folder, EXPORT_PATH, action, str(sequence)))
            except:
                pass

# Path for exported data, numpy arrays
EXPORT_PATH = os.path.join(os.path.dirname(__file__), 'MP_KeyPoints_Data') 

# # Actions that we try to detect - Default List
# ACTIONS = np.array(['Hello', 'Yes', 'No', 'Please', 'Thankyou', 'Help', 'Sorry', 'Stop', 'Done', 'More', 'Again', 'TakeCare', 'Goodbye'])

# # Actions that we try to detect - Set List
# ACTIONS = np.array(['Hello', 'Please', 'Sorry', 'Help', 'Stop', 'Thankyou', 'Done', 'More', 'Again', 'TakeCare', 'Goodbye', 'Yes', 'No'])

# Actions that we try to detect - Set List
ACTIONS = np.array(['Hello', 'Please', 'Sorry', 'Help', 'Stop', 'Thankyou', 'Done', 'More', 'Again', 'TakeCare', 'Goodbye'])

# # Actions that we try to detect - Set List
# ACTIONS = np.array(['Hello', 'Please', 'Sorry', 'Help', 'Stop', 'Thankyou', 'Done', 'More', 'Again', 'TakeCare'])

# Number of batches (groups of 20 videos each)
NUM_BATCHES = 3

# Number of sequences (videos) per batch
BATCH_SIZE = 20

# Total Number of sequences (videos) to be recorded
NUM_SEQUENCES = NUM_BATCHES * BATCH_SIZE

# Length of each sequence (number of frames per video)
SEQUENCE_LENGTH = 30

# Method call to create data folders
create_data_folders(ACTIONS, NUM_SEQUENCES, EXPORT_PATH)