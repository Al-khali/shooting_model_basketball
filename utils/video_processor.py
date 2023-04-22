import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, frame_height, frame_width, threshold):
        """
        Initialise le VideoProcessor avec la hauteur et la largeur des images des vidéos, et le seuil pour détecter les mouvements.

        :param frame_height: La hauteur des images des vidéos.
        :param frame_width: La largeur des images des vidéos.
        :param threshold: Le seuil pour détecter les mouvements dans les images des vidéos.
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.threshold = threshold

    def process_video(self, video):
        """
        Traite la vidéo pour extraire des informations utiles sur la technique de tir au basketball.

        :param video: La vidéo à traiter.
        :return: Un dictionnaire contenant des informations sur la technique de tir.
        """
        # Prétraitement de la vidéo
        preprocessed_video = self._preprocess_video(video)

        # Détection des mouvements
        movement_frames = self._detect_movement(preprocessed_video)

        # Extraction des informations sur la technique de tir
        shooting_info = self._extract_shooting_info(movement_frames)

        return shooting_info

    def _preprocess_video(self, video):
        """
        Prétraite la vidéo pour la détection des mouvements.

        :param video: La vidéo à prétraiter.
        :return: Un tableau de frames prétraitées.
        """
        frames = []
        for frame in video:
            resized_frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)
        return np.array(frames)

    def _detect_movement(self, preprocessed_video):
        """
        Détecte les mouvements dans la vidéo.

        :param preprocessed_video: La vidéo prétraitée.
        :return: Un tableau de frames contenant les mouvements détectés.
        """
        movement_frames = []
        for i in range(len(preprocessed_video) - 1):
            diff = cv2.absdiff(preprocessed_video[i], preprocessed_video[i+1])
            _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
            movement_frames.append(thresh)
        return np.array(movement_frames)

    def _extract_shooting_info(self, movement_frames):
        """
        Extrait les informations sur la technique de tir à partir des mouvements détectés.

        :param movement_frames: Un tableau de frames contenant les mouvements détectés.
        :return: Un dictionnaire contenant des informations sur la technique de tir.
        """
        shooting_info = {}

        # Calcul de la durée du shoot
        shooting_info['duration'] = len(movement_frames)

        # Calcul du nombre de mouvements pendant le shoot
        num_movements = 0
        for frame in movement_frames:
            num_movements += np.sum(frame > 0)
        shooting_info['num_movements'] = num_movements

        # Calcul de la position du shoot
        shooting_info['position'] = self._detect_position(movement_frames)

        return shooting_info

    def _detect_position(self, movement_frames):
        """
        Détecte la position du shoot à partir des mouvements détectés.

        :param movement_frames: Un

