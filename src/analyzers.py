import numpy as np

# Example function to calculate velocity and acceleration
def calculate_movement_metrics(nose_coordinates):
    velocities = []
    accelerations = []
    
    for i in range(1, len(nose_coordinates)):
        # Calculate velocity as the magnitude of the movement vector
        vector = np.array(nose_coordinates[i]) - np.array(nose_coordinates[i - 1])
        velocity = np.linalg.norm(vector)
        velocities.append(velocity)
        
        if len(velocities) > 1:
            # Calculate acceleration as the change in velocity
            acceleration = abs(velocities[-1] - velocities[-2])
            accelerations.append(acceleration)
    
    return velocities, accelerations

# Function to analyze motion based on the buffer
def analyze_motion(frameDataBuffer):
    # Extract nose coordinates from the buffer
    nose_coordinates = [data["nose2d"] for data in frameDataBuffer]
    
    if len(nose_coordinates) < 2:
        return False  # Not enough data to analyze motion

    # Calculate movement metrics
    velocities, accelerations = calculate_movement_metrics(nose_coordinates)
    
    # Define thresholds for detecting unnatural motion
    velocity_threshold = 25.0
    acceleration_threshold = 40.0

    # Check for abrupt changes
    if any(v > velocity_threshold for v in velocities) or \
       any(a > acceleration_threshold for a in accelerations):
        return False  # Unnatural motion detected
    
    return True  # Motion appears to be natural

def analyze_pose_match(frameDataBuffer, match_threshold=0.5):
    """
    Analyze if the requested direction was matched with the detected direction.
    
    :param frameDataBuffer: List of dictionaries containing frame data
    :param match_threshold: The threshold percentage (between 0 and 1) for matching success
    :return: True if the requested direction is matched as per the threshold, False otherwise
    """
    if not frameDataBuffer:
        return False  # No data to analyze
    
    # Retrieve the requested direction from the first frame
    requested_direction = frameDataBuffer[0]["requested_direction"]
    
    # Count matches
    match_count = sum(1 for data in frameDataBuffer if data["detected_direction"] == requested_direction)
    
    # Calculate match percentage
    match_percentage = match_count / len(frameDataBuffer)
    
    # Determine if the match percentage meets or exceeds the threshold
    return match_percentage >= match_threshold

def analyze_single_face(frameDataBuffer, threshold=0.05):
    """
    Validate if the number of faces detected is mostly one in all frames of the sequence.
    
    :param frameDataBuffer: List of dictionaries containing frame data
    :param threshold: Maximum allowable percentage of frames with face count other than one (e.g., 0.1 means 10%)
    :return: True if the number of faces detected is within the allowed threshold, False otherwise
    """
    if not frameDataBuffer:
        return False  # No data to analyze
    
    total_frames = len(frameDataBuffer)
    if total_frames == 0:
        return False
    
    # Count frames where number of faces detected is not equal to one
    non_single_face_count = sum(1 for data in frameDataBuffer if data["number_of_faces_detected"] != 1)
    
    # Calculate percentage of frames with non-single face count
    non_single_face_percentage = non_single_face_count / total_frames
    
    # Determine if the percentage of frames with non-single face count is within the threshold
    return non_single_face_percentage <= threshold
