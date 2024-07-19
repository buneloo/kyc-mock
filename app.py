import cv2
import imutils
import threading
from queue import Queue
from flask import Flask, render_template, Response, request, redirect, url_for
from src.faceReco import SFace
from src.faceDetYunet import FaceDetYunet
from src.faceMesh import FaceMeshMP
from src.randomSeqGenerator import generate_random_sequence
from src.analyzers import analyze_motion, analyze_pose_match, analyze_single_face
import numpy as np

import time

app = Flask(__name__)

# Initialize camera and frame queue globally
camera = cv2.VideoCapture(0)

frame_queue = Queue(maxsize=90)  # Adjust maxsize as needed

face_det = FaceDetYunet()
face_mesh = FaceMeshMP()
face_reco = SFace(key_file='data/key.key', profile_file='data/encrypted_profiles.dat')
 
liveCheck = False # flag for turning on/off the live checking mechanism

liveSequence = generate_random_sequence(15)

frameCount = 0

currentLivenessDirection = None

frameDataBuffer = []
scorePerPose = []

faceFeatures = []

def reset_data():
    global camera, frame_queue, face_det, face_mesh, face_reco
    global liveCheck, liveSequence, frameCount, currentLivenessDirection
    global frameDataBuffer, scorePerPose, faceFeatures
    
    # Reinitialize camera
    camera.release()  # Release the previous camera capture
    camera = cv2.VideoCapture(0)  # Reinitialize camera
    
    # Reinitialize frame queue
    frame_queue = Queue(maxsize=90)  # Adjust maxsize as needed
    
    # Reset liveCheck flag
    liveCheck = False  # flag for turning on/off the live checking mechanism
    
    # Generate a new random sequence
    liveSequence = generate_random_sequence(15)
    
    # Reset frame counter
    frameCount = 0
    
    # Reset current liveness direction
    currentLivenessDirection = None
    
    # Clear frame data buffer
    frameDataBuffer = []
    
    # Clear score per pose list
    scorePerPose = []
    
    # Clear face features list
    faceFeatures = []

    print("Data has been reset.")

def update_liveness_direction():
    global currentLivenessDirection, liveCheck
    while liveSequence:
        command = liveSequence.pop(0)
        currentLivenessDirection, duration = command
        time.sleep(duration)
        check_liveness()

    liveCheck = False
    currentLivenessDirection = None    

# Function to continuously capture frames
def capture_frames():
    global camera, frame_queue, frameCount
    while True:
        success, frame = camera.read()
        
        if not success:
            continue
        else:
            frame = imutils.resize(frame, width=1000)  # Resize frame for faster processing
            frameCount = frameCount + 1
            if not frame_queue.full():
                frame_queue.put(frame)

# Start frame capture in a separate thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# Function to check liveness from video stream
def check_liveness():
    global frameDataBuffer
    
    is_natural = analyze_motion(frameDataBuffer)
    print(f"Is the motion natural? {'Yes' if is_natural else 'No'}")

    is_pose_matched = analyze_pose_match(frameDataBuffer)
    print(f"Is the pose matched? {'Yes' if is_pose_matched else 'No'}")
    
    is_one_face = analyze_single_face(frameDataBuffer)
    print(f"Is the one face? {'Yes' if is_one_face else 'No'}")

    scorePerPose.append((currentLivenessDirection.value, [is_natural, is_pose_matched, is_one_face]))

    frameDataBuffer.clear()
    return True

def calculate_score():
    """
    Calculate the final score to determine if the sequence is a spoof or not,
    focusing on the presence of one face, natural motion, and pose matching.
    
    :param data: List of tuples where each tuple contains:
                 - The head pose (e.g., 'right', 'down')
                 - A list with boolean values indicating:
                   - Is the motion natural
                   - Is the pose matched
                   - Is there only one face detected
    :return: The final score and whether it's a spoof or not
    """
    # Define weights based on priority
    weight_one_face = 0.33
    weight_natural_motion = 0.33
    weight_pose_matched = 0.33

    total_frames = 0
    
    # Initialize counts
    one_face_count = 0
    natural_motion_count = 0
    pose_matched_count = 0

    # Process data
    for pose, (is_natural, is_matched, is_one_face) in scorePerPose:
        if pose == 'straight':
            continue  # Skip 'straight' pose

        total_frames += 1

        if is_one_face:
            one_face_count += 1
        if is_natural:
            natural_motion_count += 1
        if is_matched:
            pose_matched_count += 1

    # Calculate percentages
    one_face_percentage = one_face_count / total_frames
    natural_motion_percentage = natural_motion_count / total_frames
    pose_matched_percentage = pose_matched_count / total_frames
    
    # Calculate score
    score = (
        weight_one_face * one_face_percentage +
        weight_natural_motion * natural_motion_percentage +
        weight_pose_matched * pose_matched_percentage
    )

    # Determine if it is a spoof based on a threshold
    spoof_threshold = 0.85  # Define a threshold for spoof determination
    is_spoof = score < spoof_threshold

    return score, is_spoof

# Route for index page (formerly landing page)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    global liveCheck, faceFeatures
    if request.method == 'POST':
        print("login POST request received")
        liveCheck = True

        thread = threading.Thread(target=update_liveness_direction)
        thread.start()

        # Polling for liveCheck to be False
        while liveCheck:
            time.sleep(1)

        score, isSpoof = calculate_score()
        print(f"Score: {score}, Is Spoof: {isSpoof}")

        if isSpoof:
            reset_data()
            return redirect(url_for('result', message="Gotcha spoooofer!", message_type="error"))
        else:
            ret, user = face_reco.match(faceFeatures)
            reset_data()
            if ret:
                return redirect(url_for('result', message="Login successful! Welcome, " + user[0], message_type="success"))
            else:
                return redirect(url_for('result', message="Login unsuccessful! User not recognized!", message_type="error"))

    return render_template('login.html', video_feed=url_for('video_feed'))

@app.route('/result')
def result():
    message = request.args.get('message', '')
    message_type = request.args.get('message_type', 'success')
    return render_template('result.html', message=message, message_type=message_type)


@app.route('/register', methods=['GET', 'POST'])
def register():
    global liveCheck, faceFeatures
    if request.method == 'POST':
        username = request.form.get('username')  # Get username input from form
        if not username:
            return redirect(url_for('result', message="Please enter your name.", message_type="error"))

        liveCheck = True

        thread = threading.Thread(target=update_liveness_direction)
        thread.start()

        # Polling for liveCheck to be False
        while liveCheck:
            time.sleep(1)

        score, isSpoof = calculate_score()
        print(f"Score: {score}, Is Spoof: {isSpoof}")

        if isSpoof:
            reset_data()
            return redirect(url_for('result', message="Gotcha spoooofer!", message_type="error"))
        else:
            face_reco.saveProfile(username, faceFeatures)
            reset_data()
            return redirect(url_for('result', message="Registration successful!", message_type="success"))
        
        

    return render_template('register.html', video_feed=url_for('video_feed'))

# Generator function to stream video feed
def gen_frames():
    global frame_queue, liveCheck, faceFeatures
    while True:
        try:
            frame = frame_queue.get(timeout=1)  # Get the current frame from queue
            frame = cv2.flip(frame,1) # flip for selfie view

            # print("live check ---- ", liveCheck)

            if liveCheck:

                height, width, _ = frame.shape
                face_det.setImSize(width, height)

                headPose, nose2d = face_mesh.getHeadDirectionAndNose(frame)
                detectedFaces = face_det.getFaces(frame)
                noFaces = len(detectedFaces)

                if len(faceFeatures)==0 and (headPose.value == "straight") and (noFaces == 1):                    
                    faceFeatures = face_reco.recognize(frame, detectedFaces[0],)
                     
                frame_data = {
                    "frame_number": frameCount,
                    "number_of_faces_detected": noFaces,
                    "requested_direction": currentLivenessDirection.value,
                    "detected_direction": headPose.value,
                    "nose2d": nose2d
                }

                # print(frame_data)

                frameDataBuffer.append(frame_data)

            # Generate oval mask
            mask = np.zeros_like(frame)
            height, width, _ = frame.shape
            center_x, center_y = width // 2, height // 2
            axes_length_x, axes_length_y = width // 6, height // 3
            color = (255, 255, 255)  # White color for the oval (fully opaque)
            cv2.ellipse(mask, (center_x, center_y), (axes_length_x, axes_length_y), 0, 0, 360, color, -1)

            # Overlay oval mask onto the frame
            alpha = 0.3  # Adjust transparency level as needed
            blended = cv2.addWeighted(mask, alpha, frame, 1 - alpha, 0)

            # Calculate cropping dimensions
            crop_width = int(2.2 * axes_length_x)  # Adjust cropping width (e.g., 20% wider than oval)
            crop_height = int(2.2 * axes_length_y)  # Adjust cropping height (e.g., 20% wider than oval)
            start_x = max(0, center_x - crop_width // 2)
            start_y = max(0, center_y - crop_height // 2)
            end_x = min(width, center_x + crop_width // 2)
            end_y = min(height, center_y + crop_height // 2)

            # Crop the frame
            cropped_frame = blended[start_y:end_y, start_x:end_x]

            banner_height = 50  # Height of the black banner
            if currentLivenessDirection is None:
                text = ""
            else:    
                text = "Look " + currentLivenessDirection.value  # Text to display on the banner

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2

            # Get the dimensions of the cropped frame
            height, width, _ = cropped_frame.shape

            # Create a black banner image
            banner = np.zeros((banner_height, width, 3), dtype=np.uint8)
            banner[:] = (0, 0, 0)  # Black color for the banner background

            # Add text to the banner
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = banner_height - (banner_height - text_size[1]) // 2
            cv2.putText(banner, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            # Stack the cropped frame and the banner vertically
            final_frame = np.vstack((cropped_frame, banner))

            # Use your facial recognition algorithm here
            frame 
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', final_frame)
            final_frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + final_frame + b'\r\n')
            
        except Exception as e:
            print(f"Error processing frame: {e}")

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
