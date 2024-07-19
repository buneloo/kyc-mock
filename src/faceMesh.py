import numpy as np
import cv2
import mediapipe as mp
from src.utils import Direction

class FaceMeshMP():
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)

    def getHeadDirectionAndNose(self, frame):
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #flipped for selfie view
        results = self.face_mesh.process(image)

        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    
        img_h , img_w, img_c = image.shape
        face_2d = []
        face_3d = []
    
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                        if idx ==1:
                            nose_2d = (int(lm.x * img_w),int(lm.y * img_h))
                        x,y = int(lm.x * img_w),int(lm.y * img_h)
    
                        face_2d.append([x,y])
                        face_3d.append(([x,y,lm.z]))
    
                #Get 2d Coord
                face_2d = np.array(face_2d,dtype=np.float64)
    
                face_3d = np.array(face_3d,dtype=np.float64)
    
                focal_length = 1 * img_w
    
                cam_matrix = np.array([[focal_length,0,img_h/2],
                                      [0,focal_length,img_w/2],
                                      [0,0,1]])
                distortion_matrix = np.zeros((4,1),dtype=np.float64)
    
                success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)
    
                #getting rotational of face
                rmat,jac = cv2.Rodrigues(rotation_vec)
    
                angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)
    
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
    
                #here based on axis rot angle is calculated
                if y < -5:
                    ret=Direction.LEFT
                elif y > 5:
                    ret=Direction.RIGHT
                elif x < -5:
                    ret=Direction.DOWN
                elif x > 5:
                    ret=Direction.UP
                else:
                    ret=Direction.STRAIGHT

        return ret, nose_2d
