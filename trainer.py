import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getImagesWithID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L');
        faceNp = np.array(faceImg,'uint8')
        ID = int(os.path.split(imagePath) [-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
    return faces,IDs

faces,ids  = getImagesWithID('imagesdb')
recognizer.train(faces,np.array(ids))
recognizer.save('trainingData.yml')


