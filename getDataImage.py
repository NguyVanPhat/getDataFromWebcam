import cv2
import numpy as np
import os
from PIL import Image

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

path = 'dataRaw'

def getImageWithId(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    for imagePath in imagePaths:
        img = cv2.imread(imagePath)
        imgName = imagePath.split('\\')[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in boxes:
            cv2.imwrite('dataSet/' + imgName, gray[y:y + h, x:x + w])
        cv2.imshow('Trainning', img)
        cv2.waitKey(10)
getImageWithId(path)

