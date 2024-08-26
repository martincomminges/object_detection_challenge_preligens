Projet d'Inférence YOLO avec Surveillance des Ressources Système
Ce projet implémente un modèle YOLO (You Only Look Once) pour effectuer des inférences sur des vidéos capturées par des drones, tout en surveillant l'utilisation des ressources système telles que le CPU, la mémoire et la consommation d'énergie. Le projet utilise poetry pour la gestion des dépendances et l'organisation de l'environnement.

Prérequis
Avant de commencer, assurez-vous que les éléments suivants sont installés sur votre machine :

Python 3.8 ou supérieur
pip (Python package installer)
pipx (outil pour installer et exécuter des applications Python dans des environnements isolés)
Installation de Poetry avec pipx
Poetry est un gestionnaire de dépendances pour Python qui facilite la gestion des dépendances du projet et des environnements virtuels.

Installer pipx
Si pipx n'est pas encore installé, installez-le via pip :

bash
Copier le code
pip install --user pipx
pipx ensurepath
Note : Vous devrez peut-être redémarrer votre terminal ou exécuter la commande source ~/.bashrc (ou l'équivalent pour votre shell) pour que le chemin de pipx soit correctement configuré.

Installer Poetry
Utilisez pipx pour installer poetry de manière isolée :

bash
Copier le code
pipx install poetry
Vérifiez que poetry est correctement installé :

bash
Copier le code
poetry --version
Installation du Projet
Après avoir installé poetry, clonez le projet et installez ses dépendances.

Cloner le dépôt
Clonez le dépôt GitHub sur votre machine locale :

bash
Copier le code
git clone https://github.com/votre-utilisateur/votre-projet.git
cd votre-projet
Installer les Dépendances avec Poetry
Utilisez poetry pour installer toutes les dépendances nécessaires dans un environnement virtuel isolé :

bash
Copier le code
poetry install
Cette commande lit le fichier pyproject.toml et installe toutes les dépendances listées.

Exécution des Scripts en CLI
Une fois le projet installé, vous pouvez utiliser poetry pour exécuter les scripts Python.

Script d'Entraînement
Pour exécuter le script d'entraînement du modèle YOLO :

bash
Copier le code
poetry run python ./object_detection_challenge_preligens/train.py --data "./data_voc_to_yml/UAVOD_yolo/UAVOD.yaml" --model "yolov8s.pt" --save_dir "./object_detection_challenge_preligens/model" --epochs 100 --img_size 416 --device "cuda"

Script d'Inférence
Pour exécuter le script d'inférence sur une vidéo de drone :

bash
Copier le code
poetry run ppython ./object_detection_challenge_preligens/inference.py --model "./object_detection_challenge_preligens/model/yolov8s_pretrained_UAVOD-10_100.pt" --input-video "./data_voc_to_yml/testing_video/1263198-sd_640_360_30fps.mp4" --device 'cpu' --imgsz 640 --conf 0.5 --save-results --output-file './object_detection_challenge_preligens/output/output_results.txt'