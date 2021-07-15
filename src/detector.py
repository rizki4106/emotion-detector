from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import os

class DetectEmotion:
    
    __detected_face = []
    __haarcascade_path = ''
    __model_path = ''

    def __init__(self, gui=False):
        self.gui = gui
        self.__path = os.path.abspath('')
        self.__haarcascade_path = os.path.join(self.__path, "feature/haarcascade.xml")
        self.__model_path = os.path.join(self.__path, "model/Emotion_Model")

    def __image_processing(self, media=""):
        """
        calculate emotion
        """
        # read media
        self.__img = cv2.imread(os.path.join(self.__path, media))
        gray_img = cv2.cvtColor(self.__img, cv2.COLOR_BGR2GRAY)

        # initialize face model
        alg = cv2.CascadeClassifier(self.__haarcascade_path)
        wajah = alg.detectMultiScale(gray_img, scaleFactor=2.0, minNeighbors=4)

        # initial neural network model
        model = load_model(os.path.join(self.__path, self.__model_path))

        # define emotion classes
        classes = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "suprised"]

        # predicted value
        result = []

        # create rectangle
        for x, y, w, h in wajah:

            self.__img = cv2.rectangle(self.__img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # crop face
            crop = cv2.cvtColor(cv2.resize(self.__img[y:y+h, x:x+w], (48, 48), 1), cv2.COLOR_BGR2GRAY).tolist()

            # detect emotion classes
            faces = np.array([crop]).reshape(len([crop]), 48, 48, 1)
            pred = np.argmax(model.predict(faces[0:1]), axis=1)
            emotion = classes[pred[0]]
            result.append(emotion)

            # write text
            self.__img = cv2.putText(self.__img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self.__detected_face.append(crop)

        if self.gui:

            # resize image
            img_resized = cv2.resize(self.__img, (700, 500))
            # show result
            cv2.imshow("result", img_resized)
            cv2.waitKey(0)

        else:
            return result


    def predict(self, file : str, media_type : str):
        """
        predict emotion, this method take 1 argument
        1. file -> file path
        2. media_type -> image or video
        """

        if media_type == "image":
            return self.__image_processing(file)
        else:
            pass
