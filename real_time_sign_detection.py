import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import utils
import pyttsx3
import threading

# Initialize the Text-to-Speech engine
engine = pyttsx3.init()
engine_lock = threading.Lock()  # Lock to prevent concurrent access to the engine

# Function to speak a word asynchronously
def speak_word(word):
    with engine_lock:
        engine.say(word)
        engine.runAndWait()

# Global button parameters
button_width = 30
button_height = 30
button_margin = 10
sound_on = False

def draw_sentence(image, sentence):
    """
    Draws the recognized sentence at the bottom of the frame.

    Args:
        image: The input image.
        sentence: The recognized sentence to be drawn.

    Returns:
        The image with the recognized sentence drawn.
    """
    # Assuming your image height is 480 pixels
    image_height = image.shape[0]

    # Define the subtitle rectangle and text properties
    subtitle_height = 40  # Adjust the height of the black background
    subtitle_rect = ((0, image_height - subtitle_height), (image.shape[1], image_height))
    subtitle_color = (0, 0, 0)  # Black background
    text_color = (255, 255, 255)  # White text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7  # Increased font scale for larger text
    font_thickness = 2  # Increased font thickness for better visibility
    line_type = cv2.LINE_AA

    # Draw the subtitle rectangle
    cv2.rectangle(image, subtitle_rect[0], subtitle_rect[1], subtitle_color, -1)

    # Calculate text size for better positioning
    text_size = cv2.getTextSize(' '.join(sentence), font, font_scale, font_thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2  # Center the text horizontally
    text_y = subtitle_rect[0][1] + (subtitle_rect[1][1] - subtitle_rect[0][1]) // 2 + text_size[1] // 2  # Center vertically

    # Display the text at the bottom
    cv2.putText(image, ' '.join(sentence), (text_x, text_y),
                font, font_scale, text_color, font_thickness, line_type)

    return image

def toggle_sound(event, x, y, flags, param):
    """
    Toggles the sound on/off when the mute button is clicked.

    Args:
        event: The type of mouse event.
        x: The x-coordinate of the mouse click.
        y: The y-coordinate of the mouse click.
        flags: Additional flags from OpenCV for the mouse event.
        param: Additional parameters passed to the function.

    Returns:
        None
    """
    global sound_on
    # Check if the click event is within the mute button's region
    if event == cv2.EVENT_LBUTTONDOWN and \
       button_x <= x < button_x + button_width and \
       button_y <= y < button_y + button_height:
        sound_on = not sound_on

def draw_button(image, sound_on_image, sound_off_image):
    """
    Draws the sound control button on the provided image.

    Args:
        image: The input image.
        sound_on_image: The image representing the sound on state.
        sound_off_image: The image representing the sound off state.

    Returns:
        None
    """
    global button_x, button_y
    button_x = image.shape[1] - button_width - button_margin  # Adjust to position horizontally from right
    button_y = image.shape[0] - button_height - button_margin  # Adjust to position vertically from bottom
    sound_button_image = sound_on_image if sound_on else sound_off_image
    image[button_y:button_y+button_height, button_x:button_x+button_width] = sound_button_image

# Load sound control button images
sound_on_image = cv2.imread(os.path.join(os.path.dirname(__file__), "Images/soundOn.jpg"))
sound_off_image = cv2.imread(os.path.join(os.path.dirname(__file__), "Images/soundOff.jpg"))

# Resize button images
sound_on_image = cv2.resize(sound_on_image, (button_width, button_height))
sound_off_image = cv2.resize(sound_off_image, (button_width, button_height))

# Initialize variables
keypoints_sequence = []
recognized_sentence = []
threshold_confidence = 0.9

# Load the model
model_directory = os.path.join(os.path.dirname(__file__), 'Saved_Model')
model_path = os.path.join(model_directory, 'gesture_recognition_model.keras')
gesture_model = load_model(model_path)

# Initialize camera capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Holistic model
holistic_model = utils.holistic_model.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

with holistic_model as holistic:
    while cap.isOpened():
        # Read frame from camera feed
        ret, frame = cap.read()
        if not ret:
            break

        # Make detections
        image, results = utils.detect_keypoints(frame, holistic)

        # Draw landmarks
        utils.draw_landmarks(image, results)

        # Prediction logic
        frame_keypoints = utils.extract_keypoints(results)
        keypoints_sequence.append(frame_keypoints)
        keypoints_sequence = keypoints_sequence[-30:]

        # Checks if 30 frames have passed and if there are any keypoints detected in the frames to proceed with the prediction
        if len(keypoints_sequence) == 30 and not all(np.all(kp == 0) for kp in keypoints_sequence):
            prediction = gesture_model.predict(np.expand_dims(keypoints_sequence, axis=0))[0]
            predicted_action = utils.ACTIONS[np.argmax(prediction)]

            # Update recognized sentence based on prediction
            if len(recognized_sentence) > 0 and predicted_action != recognized_sentence[-1]:
                if prediction[np.argmax(prediction)] > threshold_confidence:
                    recognized_sentence.append(predicted_action)

                    # Speak the newly predicted word asynchronously if sound is on
                    if sound_on:
                        threading.Thread(target=speak_word, args=(predicted_action,)).start()

            elif len(recognized_sentence) == 0:
                recognized_sentence.append(predicted_action)

                # Speak the newly predicted word asynchronously if sound is on
                if sound_on:
                    threading.Thread(target=speak_word, args=(predicted_action,)).start()

            # Limit the recognized sentence length to 5 words
            if len(recognized_sentence) > 5:
                recognized_sentence = recognized_sentence[-5:]

        # Convert the recognized sentence to text
        caption_text = ' '.join(recognized_sentence)

        # Draw the recognized sentence at the bottom of the frame
        image = draw_sentence(image, recognized_sentence)

        # Draw sound control button
        draw_button(image, sound_on_image, sound_off_image)

        # Show frame with detections
        cv2.imshow('Sign Language Interpreter', image)

        # Set mouse callback to handle button clicks
        cv2.setMouseCallback('Sign Language Interpreter', toggle_sound)

        # Check for 'x' key event to exit the window
        if cv2.waitKey(1) & 0xFF == ord('x') or cv2.waitKey(1) & 0xFF == ord('X'):
            break

        # Check if the window is closed
        if cv2.getWindowProperty('Sign Language Interpreter', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
