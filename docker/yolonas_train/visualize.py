import supervision as sv
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def plot_detections(source_image_path, result, savename='prediction_vis'):
    image = cv2.imread(source_image_path)

    detections = sv.Detections(
        xyxy=result.prediction.bboxes_xyxy,
        confidence=result.prediction.confidence,
        class_id=result.prediction.labels.astype(int)
    )

    box_annotator = sv.BoxAnnotator()

    labels = [
        f"{result.class_names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections
    ]

    annotated_frame = box_annotator.annotate(
        scene=image.copy(),
        detections=detections,
        labels=labels
    )
    fig, ax = plt.subplots(1)
    ax.grid('off')
    ax.imshow(annotated_frame)
    Path("./visualizations/").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"./visualizations/{savename}.jpg")
    
    