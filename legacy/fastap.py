from fastapi import FastAPI, File, UploadFile
from data_preprocessor import DataPreprocessor
from cnn_model import create_cnn_model
from ml_model import create_ml_model
from video_processor import VideoProcessor
from starlette.responses import HTMLResponse
from PIL import Image
import io

app = FastAPI()

# Charger les modèles
cnn_model = create_cnn_model(input_shape=(224, 224, 3), num_classes=2)
cnn_model.load_weights('cnn_model_weights.h5')

ml_model = create_ml_model()
ml_model.load_weights('ml_model_weights.h5')

# Créer un DataPreprocessor pour la prétraitement des images
data_preprocessor = DataPreprocessor(frame_height=224, frame_width=224)

# Créer un VideoProcessor pour extraire des informations utiles sur la technique de tir
video_processor = VideoProcessor(frame_height=224, frame_width=224, threshold=30)

# Page d'accueil
@app.get('/')
def home():
    return HTMLResponse('<h1>Application de prédiction de tir au basketball</h1>')

# Endpoint pour prédire si un tir est réussi ou manqué à partir d'une vidéo
@app.post('/predict')
async def predict(file: UploadFile):
    # Lire la vidéo à partir du fichier uploadé
    contents = await file.read()
    video = io.BytesIO(contents)

    # Charger la vidéo et extraire des informations sur la technique de tir
    data = [np.array(Image.open(video))]
    X = data_preprocessor.preprocess_data(data)
    shooting_info = video_processor.process_video(X[0])

    # Utiliser les modèles pour prédire si le tir est réussi ou manqué
    cnn_prediction = cnn_model.predict(X)
    ml_prediction = ml_model.predict(X.reshape(X.shape[0], -1))

    # Calculer le score du tir
    score = cnn_prediction[0][1] * 100

    # Déterminer si le tir est réussi ou manqué
    if ml_prediction[0] == 0:
        result = 'manqué'
    else:
        result = 'réussi'

    # Afficher le résultat
    return {'result': result, 'score': score, 'shooting_info': shooting_info}

