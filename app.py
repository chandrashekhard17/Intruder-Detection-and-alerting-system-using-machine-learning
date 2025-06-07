from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import face_recognition
import os
import base64
from datetime import datetime
import time
import telegram
import asyncio
import logging

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_BOT_TOKEN = '8045825482:AAHyVMIqXjBh7rqlvgdFy4WQ_UNFAUs1KyI'
TELEGRAM_CHAT_ID = '2146005820'
RECORDING_DURATION = 10  # seconds

# Initialize Flask and Telegram
app = Flask(__name__)
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

# Create necessary directories
for directory in ['known_faces', 'unknown_faces', 'normal_recording', 'unknown_person_recordings']:
    os.makedirs(directory, exist_ok=True)

# Global variables
recording = False
unknown_person_detected = False
unknown_person_start_time = 10
video_writer = None
current_output_path = None

# Load known faces
def load_known_faces():
    known_faces, known_names = [], []
    for filename in os.listdir('known_faces'):
        filepath = os.path.join('known_faces', filename)
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image, model="large")
        if encodings:
            known_faces.extend(encodings)
            known_names.extend([os.path.splitext(filename)[0]] * len(encodings))
    return known_faces, known_names

known_faces, known_names = load_known_faces()

# Telegram message function
async def send_telegram_media(file_path, caption, is_video=False):
    try:
        if not os.path.exists(file_path):
            return
        with open(file_path, 'rb') as file:
            if is_video:
                await bot.send_video(chat_id=TELEGRAM_CHAT_ID, video=file, caption=caption)
            else:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=file, caption=caption)
        logger.info(f"Sent {'video' if is_video else 'image'} to Telegram: {file_path}")
    except Exception as e:
        logger.error(f"Error sending to Telegram: {e}")

# Video handling functions
def initialize_video_writer(cap, output_path):
    global video_writer
    if video_writer is not None:
        video_writer.release()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

def switch_recording(cap, is_unknown=False):
    global unknown_person_detected, unknown_person_start_time, current_output_path
    unknown_person_detected = is_unknown
    if is_unknown:
        unknown_person_start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    current_output_path = os.path.join(
        'unknown_person_recordings' if is_unknown else 'normal_recording',
        f"{'unknown_person' if is_unknown else 'normal_recording'}_{timestamp}.avi"
    )
    initialize_video_writer(cap, current_output_path)

# Face detection and recording
def save_unknown_face(frame, face_location):
    try:
        top, right, bottom, left = face_location
        padding = 20
        top, left = max(0, top - padding), max(0, left - padding)
        bottom, right = min(frame.shape[0], bottom + padding), min(frame.shape[1], right + padding)
        
        # Extract and save face image
        face_image = frame[top:bottom, left:right]
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filepath = os.path.join('unknown_faces', f"unknown_face_{timestamp}.jpg")
        cv2.imwrite(filepath, face_image)
        
        # Send to Telegram immediately using asyncio
        caption = f"🚨 Unknown person detected!\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(send_telegram_media(filepath, caption))
        loop.close()
        
        return filepath
    except Exception as e:
        logger.error(f"Error in save_unknown_face: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    if not name:
        return jsonify({"error": "Name is required"})

    if 'image' in request.files:
        file = request.files['image']
        filepath = os.path.join('known_faces', f"{name}.jpg")
        file.save(filepath)
    elif 'captured_image' in request.form:
        image_data = request.form['captured_image'].split(",")[1]
        image = base64.b64decode(image_data)
        filepath = os.path.join('known_faces', f"{name}.jpg")
        with open(filepath, "wb") as f:
            f.write(image)
    else:
        return jsonify({"error": "No image provided."})

    image = face_recognition.load_image_file(filepath)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_faces.append(encodings[0])
        known_names.append(name)
        return jsonify({"message": "Face registered successfully!"})
    else:
        os.remove(filepath)
        return jsonify({"error": "No face detected. Please try again."})

@app.route('/start-detection', methods=['GET'])
def start_detection():
    global recording, video_writer
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Failed to access webcam."})

    recording = True
    switch_recording(cap)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="large")

        unknown_person_present = False
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            best_match_distance = face_distances[best_match_index]
            
            name = "Unknown"
            color = (0, 0, 255)  # Red for unknown

            if best_match_distance < 0.6:
                name = known_names[best_match_index]
                color = (0, 255, 0)  # Green for known
            else:
                unknown_person_present = True
                # Save and send unknown face immediately
                save_unknown_face(frame, (top, right, bottom, left))

            # Draw rectangle and name (without confidence score)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if unknown_person_present and not unknown_person_detected:
            switch_recording(cap, True)
        elif unknown_person_detected and time.time() - unknown_person_start_time >= RECORDING_DURATION:
            if current_output_path and os.path.exists(current_output_path):
                caption = f"📹 Unknown person recording completed\nDuration: {RECORDING_DURATION} seconds"
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(send_telegram_media(current_output_path, caption, True))
                loop.close()
            switch_recording(cap)

        if recording and video_writer is not None:
            video_writer.write(frame)

        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    recording = False
    return jsonify({"message": "Detection finished."})

if __name__ == "__main__":
    try:
        # Send single initial notification
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text="🤖 Unknown Person Detection System is now active!"
        ))
        loop.close()
        
        # Run Flask app
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Error initializing the application: {e}")
