import cv2

class VideoProcessor:
    def __init__(self, model):
        """
        Initialise le VideoProcessor avec le modèle de machine learning.

        :param model: Le modèle de machine learning utilisé pour prédire les résultats.
        """
        self.model = model

    def process_video(self, video_path):
        """
        Traite la vidéo pour obtenir des conseils sur la technique de tir.

        :param video_path: Le chemin d'accès à la vidéo.
        :return: Des conseils pour améliorer la technique de tir.
        """
        video = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)

        preprocessed_frames = self._preprocess_frames(frames)
        predictions = self.model.predict(preprocessed_frames)
        tips = self._generate_tips(predictions)

        return tips

   
    
    def _preprocess_frames(self, frames):
        """
        Prétraite les images des frames pour l'entrée du modèle.

        :param frames: Les images des frames.
        :return: Les images prétraitées des frames.
        """
        def preprocess(frame):
            """
            Redimensionne l'image de la frame.

            :param frame: L'image de la frame.
            :return: L'image prétraitée.
            """
            resized_frame = cv2.resize(frame, (224, 224))
            return resized_frame 
        
        
        preprocessed_frames = []
        for frame in frames:
            # Prétraitement spécifique (par exemple, redimensionnement, normalisation, etc.)
            preprocessed_frame = preprocess(frame)
            preprocessed_frames.append(preprocessed_frame)

        return preprocessed_frames

    def _generate_tips(self, predictions):
        """
        Génère des conseils pour améliorer la technique de tir à partir des prédictions du modèle.

        :param predictions: Les prédictions du modèle.
        :return: Des conseils pour améliorer la technique de tir.
        """
        tips = []
        for prediction in predictions:
            if prediction == 0:
                tips.append("Concentre-toi sur la position de tes pieds pour améliorer ta stabilité.")
            elif prediction == 1:
                tips.append("Essaie de lancer plus haut pour améliorer la trajectoire de la balle.")
            else:
                tips.append("Travaille ta coordination main-oeil pour améliorer ta précision.")

        return tips
