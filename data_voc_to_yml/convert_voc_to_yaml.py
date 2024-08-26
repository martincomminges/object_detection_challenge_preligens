import os
import xml.etree.ElementTree as ET
import shutil
import random

# Répertoire contenant les fichiers XML du dataset VOC
voc_xml_dir = './data_voc_to_yml/small-weak-UVA-object-dataset/Annotations'

# Répertoire contenant les images du dataset VOC
voc_img_dir = './data_voc_to_yml/small-weak-UVA-object-dataset/JPEGImages'

# Répertoire de sortie pour le format YOLO
yolo_dir = './data_voc_to_yml/UAVOD_yolo'

# Créer le répertoire de sortie YOLO s'il n'existe pas
if not os.path.exists(yolo_dir):
    os.makedirs(yolo_dir)
    os.makedirs(os.path.join(yolo_dir, 'images'))
    os.makedirs(os.path.join(yolo_dir, 'labels'))

# Dictionnaire des classes du dataset VOC
voc_classes = {
    'building': 0, 'ship': 1, 'vehicle': 2, 'prefabricated-house': 3,
    'well': 4, 'cable-tower': 5, 'pool': 6, 'landslide': 7, 'cultivation-mesh-cage': 8,
    'quarry': 9
}
# Parcourir les fichiers XML du dataset VOC
for filename in os.listdir(voc_xml_dir):
    if filename.endswith('.xml'):
        # Charger le fichier XML
        tree = ET.parse(os.path.join(voc_xml_dir, filename))
        root = tree.getroot()

        # Récupérer le nom de l'image
        img_filename = root.find('filename').text
        img_path = os.path.join(voc_img_dir, img_filename)
        print(f"Image path: {img_path}")

        # Copier l'image dans le répertoire de sortie YOLO
        dst_img_path = os.path.join(yolo_dir, 'images', img_filename)
        shutil.copy(img_path, dst_img_path)
        print(f"Copied image to: {dst_img_path}")

        # Créer le fichier de labels YOLO
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(yolo_dir, 'labels', label_filename)
        with open(label_path, 'w') as f:
            # Parcourir les objets annotés dans le fichier XML
            for obj in root.findall('object'):
                # Récupérer la classe de l'objet
                obj_class = obj.find('name').text
                # Récupérer les coordonnées de la boîte englobante
                bbox = obj.find('bndbox')
                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)
                # Calculer les coordonnées normalisées du centre et de la taille de la boîte
                img_width = int(root.find('size/width').text)
                img_height = int(root.find('size/height').text)
                x_center = (x1 + x2) / (2 * img_width)
                y_center = (y1 + y2) / (2 * img_height)
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                # Écrire les informations dans le fichier de labels YOLO
                f.write(f"{voc_classes[obj_class]} {x_center} {y_center} {width} {height}\n")
            print(f"Created label file: {label_path}")

# Obtenir la liste des noms d'image
yolo_dir_image = os.path.join(yolo_dir, 'images')
image_filenames = [f for f in os.listdir(yolo_dir_image) if f.endswith('.jpg')]

# Diviser le dataset en ensembles d'entraînement et de validation
train_size = int(len(image_filenames) * 0.8)
train_filenames = random.sample(image_filenames, train_size)
val_filenames = [f for f in image_filenames if f not in train_filenames]

# Créer les fichiers train.txt, val.txt et classes.txt
with open(os.path.join(yolo_dir, 'train.txt'), 'w') as f:
    for filename in train_filenames:
        f.write(os.path.join(yolo_dir_image, filename) + '\n')

with open(os.path.join(yolo_dir, 'val.txt'), 'w') as f:
    for filename in val_filenames:
        f.write(os.path.join(yolo_dir_image, filename) + '\n')

with open(os.path.join(yolo_dir, 'classes.txt'), 'w') as f:
    for class_name in voc_classes.keys():
        f.write(class_name + '\n')