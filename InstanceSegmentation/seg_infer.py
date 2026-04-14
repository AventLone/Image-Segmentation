import supervision as sv
from rfdetr import RFDETRSegMedium
from rfdetr.assets.coco_classes import COCO_CLASSES


model = RFDETRSegMedium()

print(model.model)
# model.optimize_for_inference()
# # print(type(model))

# # detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)
# detections = model.predict("/home/avent/Pictures/images (1).jpeg", threshold=0.5)

# print(type(detections))

# labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

# annotated_image = sv.MaskAnnotator().annotate(detections.data["source_image"], detections)
# annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

# sv.plot_image(annotated_image)