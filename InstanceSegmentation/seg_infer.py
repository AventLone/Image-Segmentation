import supervision as sv
from rfdetr import RFDETRSegMedium
from rfdetr.assets.coco_classes import COCO_CLASSES
from PIL import Image
import cv2
import numpy as np

def visulaizeSeg(src_img: np.ndarray, detections):
    for i in range(len(detections)):
        # Extract data for this detection
        bbox = detections.xyxy[i].astype(int)
        mask = detections.mask[i]
        class_id = detections.class_id[i]
        conf = detections.confidence[i]

        # Generate a random color for each instance
        color = [int(c) for c in np.random.randint(0, 255, 3)]

        # --- Draw Mask ---
        # Create a colored overlay for the mask
        overlay = src_img.copy()
        overlay[mask] = color
        # Blend overlay with original image (alpha = 0.5)
        cv2.addWeighted(overlay, 0.5, src_img, 0.5, 0, src_img)

        # --- Draw Bounding Box ---
        cv2.rectangle(src_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # --- Draw Label ---
        label = f"Class {class_id}: {conf:.2f}"
        cv2.putText(src_img, label, (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return src_img
    

model = RFDETRSegMedium(pretrain_weights="output/checkpoint_best_ema.pth", num_queries=100, num_select=80, num_classes=2)
model.optimize_for_inference()
image = Image.open("/home/linde/Downloads/rf/valid/Replicator/0076.png").convert("RGB")
detections = model.predict(image, threshold=0.7)

# 1. Convert PIL image to OpenCV BGR format
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

image_cv = visulaizeSeg(image_cv, detections)

# 3. Display or save
cv2.imshow("RF-DETR Detections", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()