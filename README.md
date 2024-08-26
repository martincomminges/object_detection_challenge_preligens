## Prérequis
Avant de commencer, assurez-vous que les éléments suivants sont installés sur votre machine :

Python 3.8 ou supérieur
pip (Python package installer)
pipx (outil pour installer et exécuter des applications Python dans des environnements isolés)
Installation de Poetry avec pipx


Installer pipx
Si pipx n'est pas encore installé, installez-le via pip :

```
pip install --user pipx
pipx ensurepath
```


Installer Poetry
Utilisez pipx pour installer poetry de manière isolée :

```
pipx install poetry
```
Vérifiez que poetry est correctement installé :

```
poetry --version
```


## Cloner le dépôt
Clonez le dépôt GitHub sur votre machine locale :


```
git clone https://github.com/martincomminges/object_detection_challenge_preligens.git
```
```
poetry install
```


## Exécution des Scripts en CLI
Une fois le projet installé, vous pouvez utiliser poetry pour exécuter les scripts Python.

Script d'Entraînement
Pour exécuter le script d'entraînement du modèle YOLO :

```
poetry run python ./object_detection_challenge_preligens/train.py --data "./data_voc_to_yml/UAVOD_yolo/UAVOD.yaml" --model "yolov8s.pt" --save_dir "./object_detection_challenge_preligens/model" --epochs 100 --img_size 416 --device "cuda"
```

Script d'Inférence
Pour exécuter le script d'inférence sur une vidéo de drone :
```
poetry run python ./object_detection_challenge_preligens/inference.py --model "./object_detection_challenge_preligens/model/yolov8s_pretrained_UAVOD-10_100.pt" --input-video "./data_voc_to_yml/testing_video/1263198-sd_640_360_30fps.mp4" --device 'cpu' --imgsz 640 --conf 0.5 --save-results --output-file './object_detection_challenge_preligens/output/output_results.txt'
```