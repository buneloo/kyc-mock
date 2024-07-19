import cv2

class FaceDetYunet():
    def __init__(self):
        self.faceDet = cv2.FaceDetectorYN_create("models/face_detection_yunet_2023mar.onnx", "", (0,0))
        

    def setImSize(self,imW, imH):
        self.faceDet.setInputSize((imW, imH))

    def getFaces(self, frame):
        faces = []

        _, results = self.faceDet.detect(frame)    

        for face in results:
            faces.append(face)

        return faces    
            