from datetime import datetime
from rfdetr import RFDETRSegMedium

model = RFDETRSegMedium(pretrain_weights=None, num_queries=100, num_select=80, num_classes=2)
model.train(
    dataset_dir="/home/linde/Downloads/rf",
    num_workers=16,
    epochs=36, batch_size=4, grad_accum_steps=4,
    lr=1e-3, lr_scheduler='cosine',  # Enables the Cosine Annealing scheduler
    output_dir="output", progress_bar=False,
    wandb=True,                                         # Enables W&B logging
    project="Instance Segmentation",                    # Optional: Specify W&B project
    run=datetime.now().strftime("%Y-%m-%d-%H%M%S")      # Optional: Specify W&B run name
)

model.export(output_dir="exported_models")