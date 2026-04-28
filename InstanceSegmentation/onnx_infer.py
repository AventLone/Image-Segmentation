import cv2
import numpy as np
import onnxruntime as ort

# 1. Load Session
session = ort.InferenceSession("/home/avent/Desktop/CageStack_IBVS/pallet_detection_ws/inference_model.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# 2. Preprocess Image
img_orig = cv2.imread("/home/avent/Desktop/generated_data/rf/valid/Replicator/0007.png")
h_orig, w_orig = img_orig.shape[:2]
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (432, 432)) # Typical RT-DETR input size
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))[np.newaxis, :]

# 3. Run Inference
# Outputs usually: [boxes, scores, masks]
outputs = session.run(None, {input_name: img})
boxes, scores, mask_logits = outputs[0], outputs[1], outputs[2]

def visualize(image, boxes, scores, masks, threshold=0.5):
    annotated = image.copy()
    
    # Iterate through detections
    for i in range(len(boxes[0])):
        score = scores[0][i].max()
        if score < threshold:
            continue
            
        # 1. Process Bounding Box
        box = boxes[0][i] # Format usually [x_min, y_min, x_max, y_max]
        x1, y1, x2, y2 = map(int, box)
        
        # 2. Process Mask
        # RT-DETR masks are often (N, 160, 160). Upsample to original image size.
        mask = masks[0][i]
        mask = cv2.resize(mask, (w_orig, h_orig))
        mask = (mask > 0).astype(np.uint8) * 255 # Thresholding logits
        
        # 3. Draw Mask Overlay
        color = (0, 255, 0) # Green for all masks
        colored_mask = np.zeros_like(annotated, dtype=np.uint8)
        colored_mask[mask > 0] = color
        cv2.addWeighted(annotated, 1.0, colored_mask, 0.4, 0, annotated)
        
        # 4. Draw Box and Label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{scores[0][i].argmax()} {score:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated

# Final display
result_img = visualize(img_orig, boxes, scores, mask_logits)
cv2.imshow("RT-DETR Segmentation", result_img)
cv2.waitKey(0)
