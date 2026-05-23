from datetime import datetime
from rfdetr import RFDETRSegMedium, RFDETRSegLarge, RFDETRSegSmall, RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES
from rfdetr.datasets.aug_config import AUG_AGGRESSIVE

model = RFDETRMedium(num_queries=100, num_select=80, num_classes=3)
# model = RFDETRSegMedium(num_classes=3)

model_name = model.__class__.__name__
run_dix = datetime.now().strftime("%Y-%m-%d-%H%M%S")

model.train(
    dataset_dir="/home/linde/Desktop/Datasets/0518_rfdetr",
    num_workers=16,
    epochs=100, batch_size=4, grad_accum_steps=4,
    lr=6e-4, lr_scheduler='cosine',  # Enables the Cosine Annealing scheduler
    # lr=1e-4,
    # resolution=504,
    aug_config=AUG_AGGRESSIVE,
    output_dir=f"output/{model_name}/{run_dix}", progress_bar=False,
    wandb=True,                                         # Enables W&B logging
    project=model_name,                          # Optional: Specify W&B project
    run=datetime.now().strftime("%Y-%m-%d-%H%M%S")      # Optional: Specify W&B run name
)

model.export(opset_version=21, output_dir=f"exported_models/{model_name}/{run_dix}", verbose=False)