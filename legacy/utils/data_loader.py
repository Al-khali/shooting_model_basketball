import os
import cv2
import numpy as np

class DataLoader:
    def __init__(self, data_dir):
        """
        Initialise le DataLoader avec le répertoire contenant les données brutes.

        :param data_dir: Le répertoire contenant les vidéos des joueurs en train de shooter le ballon.
        """
        self.data_dir = data_dir

    def load_data(self):
        """
        Charge les données brutes à partir du répertoire spécifié.

        :return: Un tableau de vidéos (une vidéo par élément).
        """
        data = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".mp4"):
                path = os.path.join(self.data_dir, filename)
                video = cv2.VideoCapture(path)
                frames = []
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    frames.append(frame)
                data.append(np.array(frames))
        return np.array(data)
