import cv2
import numpy as np

class DataPreprocessor:
    def __init__(self, frame_height, frame_width):
        """
        Initialise le DataPreprocessor avec la hauteur et la largeur des images des vidéos.

        :param frame_height: La hauteur des images des vidéos.
        :param frame_width: La largeur des images des vidéos.
        """
        self.frame_height = frame_height
        self.frame_width = frame_width

    def preprocess_data(self, data):
        """
        Prétraite les données brutes pour l'entraînement du modèle.

        :param data: Un tableau de vidéos (une vidéo par élément).
        :return: Un tableau de données prétraitées.
        """
        preprocessed_data = []
        for video in data:
            frames = []
            for frame in video:
                resized_frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                frames.append(resized_frame)
            preprocessed_data.append(np.array(frames))
        return np.array(preprocessed_data)
