import cv2
import pickle
from cryptography.fernet import Fernet
import os

COSINE_THRESHOLD = 0.5

class SFace:
    def __init__(self, key_file='key.key', profile_file='encrypted_profiles.dat'):
        self.face_recognizer = cv2.FaceRecognizerSF_create("models/face_recognition_sface_2021dec.onnx", "")
        self.dictionary = {}
        
        self.key_file = key_file
        self.profile_file = profile_file
        
        # Attempt to load the encryption key from the file
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as key_file:
                self.key = key_file.read()
        else:
            # Generate a new key if the file does not exist
            self.key = Fernet.generate_key()
            self.save_key(self.key_file)
        
        # Initialize the cipher suite with the key
        self.cipher_suite = Fernet(self.key)
        
        # Load the profile (dictionary) from the encrypted file
        self.load_profile(self.profile_file)
    
    def match(self, faceFeat):
        max_score = 0.0
        sim_user_id = ""
        for user_id, feature2 in zip(self.dictionary.keys(), self.dictionary.values()):
            score = self.face_recognizer.match(
                faceFeat, feature2, cv2.FaceRecognizerSF_FR_COSINE)
            if score >= max_score:
                max_score = score
                sim_user_id = user_id
        if max_score < COSINE_THRESHOLD:
            return False, ("", 0.0)
        return True, (sim_user_id, max_score)
    
    def recognize(self, image, face):
        aligned_face = self.face_recognizer.alignCrop(image, face)
        feat = self.face_recognizer.feature(aligned_face)
        return feat
    
    def saveProfile(self, name, face):
        self.dictionary[name] = face
        self.save_profile(self.profile_file)
    
    def save_key(self, key_file):
        with open(key_file, 'wb') as key_file:
            key_file.write(self.key)
    
    def save_profile(self, profile_file):
        with open(profile_file, 'wb') as file:
            encrypted_data = self.cipher_suite.encrypt(pickle.dumps(self.dictionary))
            file.write(encrypted_data)
    
    def load_profile(self, profile_file):
        if os.path.exists(profile_file):
            with open(profile_file, 'rb') as file:
                encrypted_data = file.read()
                self.dictionary = pickle.loads(self.cipher_suite.decrypt(encrypted_data))
