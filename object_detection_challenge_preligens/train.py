import os
import argparse
from ultralytics import YOLO, settings

# Mise à jour des paramètres globaux
settings.update({'mlflow': True})

# Configuration des chemins
DATA_PATH = "./object_detection_challenge_preligens/data_voc_to_yml/UAVOD_yolo/UAVOD.yaml"
MODEL_PATH = "yolov8s.pt"
SAVE_DIR = "./object_detection_challenge_preligens/model"
SAVE_PATH = os.path.join(SAVE_DIR, "yolov8s_pretrained_UAVOD-10_100.pt")
DEVICE = "cuda"
EPOCHS = 100
IMG_SIZE = 416

def parse_arguments():
    """
    Fonction pour analyser les arguments de la ligne de commande.

    :return: Un objet Namespace avec tous les arguments de la CLI.
    """
    parser = argparse.ArgumentParser(description="Entraînement et validation d'un modèle YOLO pour la détection d'objets.")
    
    parser.add_argument('--data', type=str, default=DATA_PATH, required=True, help='Chemin vers le fichier YAML de configuration des données.')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Chemin vers le modèle pré-entraîné.')
    parser.add_argument('--save_dir', type=str, default='./model', help='Répertoire pour sauvegarder le modèle entraîné.')
    parser.add_argument('--epochs', type=int, default=100, help='Nombre d\'époques pour l\'entraînement.')
    parser.add_argument('--img_size', type=int, default=416, help='Taille des images pour l\'entraînement.')
    parser.add_argument('--device', type=str, default='cuda', help='Appareil à utiliser pour l\'entraînement et l\'inférence (cpu ou cuda).')

    return parser.parse_args()

def train_yolo_model(data_path, model_path, epochs, img_size, device):
    """
    Fonction pour entraîner un modèle YOLO.
    
    :param data_path: Chemin vers le fichier de configuration des données YAML.
    :param model_path: Chemin vers le modèle pré-entraîné.
    :param epochs: Nombre d'époques pour l'entraînement.
    :param img_size: Taille des images pour l'entraînement.
    :param device: Appareil à utiliser ('cpu' ou 'cuda').
    :return: Résultats de l'entraînement.
    """
    # Chargement du modèle pré-entraîné
    model = YOLO(model_path, task='detect')
    
    # Entraînement du modèle
    results = model.train(data=data_path, epochs=epochs, imgsz=img_size, device=device)
    return model, results

def save_model(model, save_path):
    """
    Fonction pour sauvegarder un modèle YOLO entraîné.
    
    :param model: Modèle YOLO entraîné.
    :param save_path: Chemin où sauvegarder le modèle.
    """
    model.save(save_path)
    print(f"Modèle sauvegardé à {save_path}")

def validate_model(model_path, data_path):
    """
    Fonction pour charger et valider un modèle YOLO sauvegardé.
    
    :param model_path: Chemin vers le modèle sauvegardé.
    :param data_path: Chemin vers le fichier de configuration des données YAML.
    :return: Résultats de la validation.
    """
    # Charger le modèle sauvegardé
    model = YOLO(model_path)
    
    # Validation du modèle
    results = model.val(data=data_path)
    return results

def main():
    # Parsing des arguments CLI
    args = parse_arguments()

    # Création du répertoire de sauvegarde si nécessaire
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    save_path = os.path.join(args.save_dir, "yolov8s_pretrained_UAVOD-10_100.pt")

    # Entraînement du modèle
    model, results_training = train_yolo_model(args.data, args.model, args.epochs, args.img_size, args.device)
    
    # Sauvegarde du modèle entraîné
    save_model(model, save_path)
    
    # Validation du modèle sauvegardé
    #results_val = validate_model(save_path, args.data)

    # Afficher les résultats de validation
    #print(f"Résultats de validation: {results_val}")

if __name__ == "__main__":
    main()

#python ./object_detection_challenge_preligens/train.py --data "./data_voc_to_yml/UAVOD_yolo/UAVOD.yaml" --model "yolov8s.pt" --save_dir "./object_detection_challenge_preligens/model" --epochs 100 --img_size 416 --device "cuda"