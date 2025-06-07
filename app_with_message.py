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
from telegram.ext import Application
import logging
from dotenv import load_dotenv
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get configuration from environment variables with fallback values
TELEGRAM_BOT_TOKEN = '8045825482:AAHyVMIqXjBh7rqlvgdFy4WQ_UNFAUs1KyI'
TELEGRAM_CHAT_ID = '2146005820'
RECORDING_DURATION = 10  # seconds

app = Flask(__name__)

# Create a thread pool for async operations
executor = ThreadPoolExecutor(max_workers=2)

# Initialize bot
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

# Directories for storing images and videos
KNOWN_FACES_DIR = "known_faces"
UPLOADS_DIR = "uploads"
NORMAL_RECORDING_DIR = "normal_recording"
UNKNOWN_PERSON_DIR = "unknown_person_recordings"
UNKNOWN_FACES_DIR = "unknown_faces"

# Create directories if they don't exist
for directory in [KNOWN_FACES_DIR, UPLOADS_DIR, NORMAL_RECORDING_DIR, 
                 UNKNOWN_PERSON_DIR, UNKNOWN_FACES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Load known faces
def load_known_faces():
    known_faces, known_names = [], []
    for filename in os.listdir(KNOWN_FACES_DIR):
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image, model="large")
        if encodings:
            known_faces.extend(encodings)
            known_names.extend([os.path.splitext(filename)[0]] * len(encodings))
    return known_faces, known_names

known_faces, known_names = load_known_faces()

# Add these global variables
recording = False
unknown_person_detected = False
unknown_person_start_time = 0
video_writer = None
current_output_path = None

def initialize_video_writer(cap, output_path):
    """Initialize video writer with proper settings"""
    global video_writer
    try:
        if video_writer is not None:
            video_writer.release()
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20.0  # Frames per second
        
        # Initialize video writer with high quality settings
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )
        logger.info(f"Initialized video writer for: {output_path}")
    except Exception as e:
        logger.error(f"Error initializing video writer: {e}")

def switch_recording(cap, is_unknown=False):
    """Switch to recording unknown person with proper timing"""
    global unknown_person_detected, unknown_person_start_time, current_output_path
    try:
        unknown_person_detected = is_unknown
        if is_unknown:
            unknown_person_start_time = time.time()
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        current_output_path = os.path.join(
            UNKNOWN_PERSON_DIR if is_unknown else NORMAL_RECORDING_DIR,
            f"{'unknown_person' if is_unknown else 'normal_recording'}_{timestamp}.avi"
        )
        initialize_video_writer(cap, current_output_path)
        
        # Send initial alert
        caption = f"🎥 Recording {'unknown person' if is_unknown else 'normal recording'}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(send_telegram_media(current_output_path, caption, is_video=is_unknown))
        loop.close()
        
    except Exception as e:
        logger.error(f"Error in switch_recording: {e}")

async def send_telegram_media(file_path, caption, is_video=False):
    """Send media to Telegram with proper error handling"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"Media file not found: {file_path}")
            return
            
        with open(file_path, 'rb') as media:
            if is_video:
                await bot.send_video(
                    chat_id=TELEGRAM_CHAT_ID,
                    video=media,
                    caption=caption,
                    supports_streaming=True
                )
            else:
                await bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=media,
                    caption=caption
                )
        logger.info(f"Successfully sent {'video' if is_video else 'image'} to Telegram: {file_path}")
    except Exception as e:
        logger.error(f"Error sending media to Telegram: {e}")

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Detection Page
@app.route('/detect')
def detect():
    return render_template('detect.html')

# Function to send Telegram message with image
async def send_telegram_image(image_path, caption):
    """Send image to Telegram with proper error handling"""
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return
            
        with open(image_path, 'rb') as photo:
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=photo,
                caption=caption
            )
        logger.info(f"Successfully sent image to Telegram: {image_path}")
    except Exception as e:
        logger.error(f"Error sending image to Telegram: {e}")

# Modified save_unknown_face function
def save_unknown_face(frame, face_location):
    """Save the detected unknown face as an image and send to Telegram"""
    try:
        top, right, bottom, left = face_location
        padding = 20
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(frame.shape[0], bottom + padding)
        right = min(frame.shape[1], right + padding)
        
        # Extract and save face image
        face_image = frame[top:bottom, left:right]
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"unknown_face_{timestamp}.jpg"
        filepath = os.path.join(UNKNOWN_FACES_DIR, filename)
        
        # Save the image
        cv2.imwrite(filepath, face_image)
        logger.info(f"Saved unknown face image: {filepath}")
        
        # Send to Telegram
        caption = f"🚨 Unknown person detected!\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Use asyncio to send the image
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(send_telegram_media(filepath, caption))
        loop.close()
        
        return filepath
    except Exception as e:
        logger.error(f"Error in save_unknown_face: {e}")
        return None

# Modified start_detection route
@app.route('/start-detection', methods=['GET'])
def start_detection():
    global recording, video_writer
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return jsonify({"error": "Failed to access webcam."})

    recording = True
    switch_recording(cap)

    FACE_DISTANCE_THRESHOLD = 0.6
    MIN_FACE_SIZE = 100

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="large")

        unknown_person_present = False
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_height = bottom - top
            face_width = right - left
            
            if face_height < MIN_FACE_SIZE or face_width < MIN_FACE_SIZE:
                continue

            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            best_match_distance = face_distances[best_match_index]
            
            name = "Unknown"
            color = (0, 0, 255)

            if best_match_distance < FACE_DISTANCE_THRESHOLD:
                name = known_names[best_match_index]
                color = (0, 255, 0)
            else:
                unknown_person_present = True
                save_unknown_face(frame, (top, right, bottom, left))

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f"{name} ({1 - best_match_distance:.2f})"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Handle unknown person detection and recording
        current_time = time.time()
        if unknown_person_present and not unknown_person_detected:
            switch_recording(cap, True)
        elif unknown_person_detected:
            elapsed_time = current_time - unknown_person_start_time
            if elapsed_time >= RECORDING_DURATION:
                # Ensure we've recorded the full duration
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                
                # Send the completed video
                if current_output_path and os.path.exists(current_output_path):
                    caption = f"📹 Unknown person recording completed\nDuration: {RECORDING_DURATION} seconds"
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(send_telegram_media(current_output_path, caption, True))
                    loop.close()
                
                switch_recording(cap)

        # Write frame to video if recording
        if recording and video_writer is not None:
            video_writer.write(frame)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    recording = False
    return jsonify({"message": "Detection finished."})

# Add new route for Telegram webhook
@app.route('/webhook', methods=['POST'])
def webhook():
    update = telegram.Update.de_json(request.get_json(), bot)
    # Handle incoming messages if needed
    return jsonify({"status": "ok"})

# Let's also add some welcome message functionality
async def send_welcome_message():
    """Send welcome message when bot starts"""
    try:
        message = (
            "👋 Welcome to Unknown Person Detection Bot!\n\n"
            "I will notify you when:\n"
            "🚨 Unknown faces are detected\n"
            "📸 Send images of unknown faces\n"
            "🎥 Send video recordings of unknown persons\n\n"
            "The system is now active and monitoring."
        )
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message
        )
    except Exception as e:
        logger.error(f"Error sending welcome message: {e}")

async def test_bot_connection():
    """Test the bot connection and send a test message"""
    try:
        message = "🤖 Bot is now active and ready to monitor for unknown faces!"
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message
        )
        logger.info("Test message sent successfully")
    except Exception as e:
        logger.error(f"Error sending test message: {e}")

# Add a function to send existing unknown face images
def send_existing_unknown_faces():
    """Send all existing unknown face images to Telegram"""
    try:
        for filename in os.listdir(UNKNOWN_FACES_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(UNKNOWN_FACES_DIR, filename)
                caption = f"📸 Previously detected unknown face: {filename}"
                
                # Use asyncio to send the image
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(send_telegram_media(filepath, caption))
                loop.close()
    except Exception as e:
        logger.error(f"Error sending existing unknown faces: {e}")

# Add function to send existing recordings
def send_existing_recordings():
    """Send all existing unknown person recordings to Telegram"""
    try:
        # Send existing unknown face images
        for filename in os.listdir(UNKNOWN_FACES_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(UNKNOWN_FACES_DIR, filename)
                caption = f"📸 Previously detected unknown face: {filename}"
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(send_telegram_media(filepath, caption))
                loop.close()

        # Send existing video recordings
        for filename in os.listdir(UNKNOWN_PERSON_DIR):
            if filename.endswith('.avi'):
                filepath = os.path.join(UNKNOWN_PERSON_DIR, filename)
                caption = f"📹 Previously recorded unknown person: {filename}"
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(send_telegram_media(filepath, caption, True))
                loop.close()
    except Exception as e:
        logger.error(f"Error sending existing recordings: {e}")

# Modify the main section to include welcome message
if __name__ == "__main__":
    try:
        # Initialize Telegram bot
        logger.info("Initializing Telegram bot...")
        
        # Send test message
        asyncio.run(bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text="🤖 Unknown Person Detection System is now active!"
        ))
        
        # Send any existing recordings
        send_existing_recordings()
        
        # Run Flask app
        logger.info("Starting Flask application...")
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Error initializing the application: {e}")
