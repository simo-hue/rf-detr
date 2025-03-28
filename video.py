import os
import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRBase()

def callback(frame, index):
    detections = model.predict(frame, threshold=0.5)
        
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)
    return annotated_frame

def process_all_videos(source_directory, target_directory):
    # Ottieni tutti i file nella directory
    for filename in os.listdir(source_directory):
        if filename.endswith(".mp4"):  # Filtro per video (modifica l'estensione se necessario)
            source_path = os.path.join(source_directory, filename)
            target_path = os.path.join(target_directory, filename)
            print(f"Processing video: {filename}")
            process_video(source_path=source_path, target_path=target_path, callback=callback)

# Chiamata alla funzione per processare tutti i video nella directory
process_all_videos(source_directory="rfdetr/assets/VIDEO/SOURCE", target_directory="rfdetr/assets/VIDEO/TARGET")