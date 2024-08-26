import time
import psutil
import subprocess
import getpass
import argparse
from scalene import scalene_profiler
from scalene.scalene_profiler import enable_profiling
from ultralytics import YOLO

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Script d'inférence YOLO avec surveillance des ressources système.")
    parser.add_argument('--model', type=str, required=True, help="Chemin vers le modèle YOLO exporté.")
    parser.add_argument('--input-video', type=str, required=True, help="Chemin vers la vidéo d'entrée.")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help="Dispositif pour exécuter le modèle ('cpu' ou 'cuda').")
    parser.add_argument('--imgsz', type=int, nargs='+', default=640, help="Taille des images pour l'inférence.")
    parser.add_argument('--conf', type=float, default=0.5, help="Seuil de confiance pour la détection des objets.")
    parser.add_argument('--sudo-password', type=str, help="Mot de passe sudo pour lancer powerstat.")
    parser.add_argument('--save-results', action='store_true', help="Enregistrer les résultats de l'inférence.")
    parser.add_argument('--output-file', type=str, default='inference_results.txt', help="Chemin du fichier de sortie pour enregistrer les résultats.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Charger le modèle sauvegardé
    exported_model = YOLO(args.model)

    # Demander le mot de passe sudo si non fourni
    sudo_password = args.sudo_password or getpass.getpass("Entrez votre mot de passe sudo : ")

    # Construire la commande sudo avec le mot de passe
    command = f'echo {sudo_password} | sudo -S powerstat -d 0.1 10'

    # Lancer powerstat en arrière-plan pour mesurer la consommation d'énergie globale
    powerstat_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Mesurer l'utilisation du CPU et de la RAM avant l'exécution du modèle
    cpu_percent_start = psutil.cpu_percent(interval=None)
    memory_info_start = psutil.virtual_memory()

    # Début du timing
    start_time = time.perf_counter()

    # Exécuter le modèle avec les paramètres spécifiés
    results_pred = exported_model(args.input_video, stream=True, imgsz=args.imgsz, device=args.device, save=args.save_results, conf=args.conf)

    # Fin du timing
    end_time = time.perf_counter()

    # Mesurer l'utilisation du CPU et de la RAM après l'exécution du modèle
    cpu_percent_end = psutil.cpu_percent(interval=None)
    memory_info_end = psutil.virtual_memory()

    # Calculer le temps d'exécution
    execution_time = end_time - start_time

    # Attendre la fin de powerstat
    powerstat_output, _ = powerstat_process.communicate()

    # Initialiser une liste pour stocker les vitesses
    speed_times = []

    # Itérer sur les résultats de l'inférence
    for result in results_pred:
        # Ajouter les temps de l'inférence au CPU
        speed_times.append(result.speed)  # 'speed' est un dictionnaire avec les temps pour différentes étapes

    # Calculer la moyenne pour chaque clé (inference, nms, etc.)
    average_speed = {key: sum(d[key] for d in speed_times) / len(speed_times) for key in speed_times[0]}

    # Enregistrer les résultats dans un fichier texte
    with open(args.output_file, 'w') as f:
        f.write("Moyenne des vitesses d'inférence :\n")
        for key, value in average_speed.items():
            f.write(f"{key}: {value:.2f} ms\n")
        f.write(f"\nTemps d'exécution : {execution_time} secondes\n")
        f.write(f"Utilisation CPU au début : {cpu_percent_start}%, à la fin : {cpu_percent_end}%\n")
        f.write(f"Utilisation de la mémoire au début : {memory_info_start.percent}%, à la fin : {memory_info_end.percent}%\n")
        f.write(f"\nRésultats de powerstat : \n{powerstat_output.decode('utf-8')}\n")

    print(f"Les résultats d'inférence et de performance ont été enregistrés dans {args.output_file}")

if __name__ == "__main__":
    scalene_profiler.start()
    
    main()
    scalene_profiler.stop()
#python inference_script.py --model-path './model/yolov8s_pretrained_UAVOD-10_100.pt' --input-video '/path/to/video.mp4' --device 'cpu' --imgsz 640 640 --conf 0.5 --save-results --output-file 'output_results.txt'
#python ./object_detection_challenge_preligens/inference.py --model "./object_detection_challenge_preligens/model/yolov8s_pretrained_UAVOD-10_100.pt" --input-video "./object_detection_challenge_preligens/data_voc_to_yml/testing_video/1263198-sd_640_360_30fps.mp4" --device 'cpu' --imgsz 640 --conf 0.5 --save-results --output-file './object_detection_challenge_preligens/output/output_results.txt'
#python ./object_detection_challenge_preligens/inference.py --model "./object_detection_challenge_preligens/model/yolov8s_pretrained_UAVOD-10_100.pt" --input-video "./data_voc_to_yml/testing_video/1263198-sd_640_360_30fps.mp4" --device 'cpu' --imgsz 640 --conf 0.5 --save-results --output-file './object_detection_challenge_preligens/output/output_results.txt'