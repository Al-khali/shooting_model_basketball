from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from cnn_model import create_cnn_model
from ml_model import create_ml_model
from video_processor import VideoProcessor
from metrics import calculate_accuracy, calculate_confusion_matrix, calculate_classification_report

# Charger les données brutes
data_loader = DataLoader(data_dir='data/raw_data')
data = data_loader.load_data()

# Prétraiter les données pour l'entraînement des modèles
data_preprocessor = DataPreprocessor(frame_height=224, frame_width=224)
X_train = data_preprocessor.preprocess_data(data)

# Créer un modèle de réseau de neurones convolutifs
cnn_model = create_cnn_model(input_shape=(224, 224, 3), num_classes=2)

# Entraîner le modèle de réseau de neurones convolutifs
y_train = np.array([0 if i < len(X_train)/2 else 1 for i in range(len(X_train))])
cnn_model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Créer un modèle de machine learning classique
ml_model = create_ml_model()

# Entraîner le modèle de machine learning classique
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
ml_model.fit(X_train_flattened, y_train)

# Charger une nouvelle vidéo à traiter
new_video = data_loader.load_data()[0]

# Traiter la vidéo pour extraire des informations sur la technique de tir
video_processor = VideoProcessor(frame_height=224, frame_width=224, threshold=30)
shooting_info = video_processor.process_video(new_video)

# Utiliser les modèles pour prédire si le tir est réussi ou manqué
X_new = data_preprocessor.preprocess_data(np.array([new_video]))
cnn_prediction = cnn_model.predict(X_new)
ml_prediction = ml_model.predict(X_new.reshape(X_new.shape[0], -1))

# Afficher les résultats
print('CNN prediction:', cnn_prediction)
print('ML prediction:', ml_prediction)
print('Shooting info:', shooting_info)

# Calculer les métriques de performance
y_true = 0 if shooting_info['position'] == 'centre' else 1
y_pred_cnn = np.argmax(cnn_prediction)
y_pred_ml = ml_prediction[0]
accuracy_cnn = calculate_accuracy([y_true], [y_pred_cnn])
accuracy_ml = calculate_accuracy([y_true], [y_pred_ml])
confusion_matrix_cnn = calculate_confusion_matrix([y_true], [y_pred_cnn])
confusion_matrix_ml = calculate_confusion_matrix([y_true], [y_pred_ml])
target_names = ['Manqué', 'Réussi']
classification_report_cnn = calculate_classification_report([y_true], [y_pred_cnn], target_names=target_names)
classification_report_ml = calculate_classification_report([y_true], [y_pred_ml], target_names=target_names)

print('CNN accuracy:', accuracy_cnn)
print('CNN confusion matrix:\n', confusion_matrix_cnn)
print('CNN classification report:\n', classification_report_cnn)

print('ML accuracy:', accuracy_ml)
print('ML confusion matrix:\n', confusion_matrix_ml)
print('ML classification report:\n', classification_report_ml)

