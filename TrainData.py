import cv2
import numpy as np
import os
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = 'dataSet'


def getImageWithId(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faces = []
    IDs = []

    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')

        faceNp = np.array(faceImage, 'uint8')

        Id = int(imagePath.split('\\')[1].split('.')[1])

        faces.append(faceNp)

        IDs.append(Id)

        cv2.imshow('Trainning', faceNp)
        cv2.waitKey(10)
    return faces, IDs


faces, Ids = getImageWithId(path)

recognizer.train(faces, np.array(Ids))

if not os.path.exists('recognizer'):
    os.makedirs('recognizer')

recognizer.save('recognizer/trainningData.yml')

cv2.destroyAllWindows()
