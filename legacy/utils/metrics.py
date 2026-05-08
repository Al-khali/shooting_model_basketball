import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def calculate_accuracy(y_true, y_pred):
    """
    Calcule la précision du modèle.

    :param y_true: Les vraies étiquettes.
    :param y_pred: Les prédictions du modèle.
    :return: La précision.
    """
    return accuracy_score(y_true, y_pred)

def calculate_confusion_matrix(y_true, y_pred):
    """
    Calcule la matrice de confusion du modèle.

    :param y_true: Les vraies étiquettes.
    :param y_pred: Les prédictions du modèle.
    :return: La matrice de confusion.
    """
    return confusion_matrix(y_true, y_pred)

def calculate_classification_report(y_true, y_pred, target_names):
    """
    Calcule le rapport de classification du modèle.

    :param y_true: Les vraies étiquettes.
    :param y_pred: Les prédictions du modèle.
    :param target_names: Les noms des classes de sortie.
    :return: Le rapport de classification.
    """
    return classification_report(y_true, y_pred, target_names=target_names)

