from datetime import datetime
from rfdetr import RFDETRSegMedium, RFDETRSegLarge, RFDETRSegSmall, RFDETRMedium
from rfdetr.datasets.aug_configs import AUG_AGGRESSIVE

model = RFDETRSegMedium(pretrain_weights="output/RFDETRSegMedium/2026.06.18-11:32/checkpoint_best_total.pth",
                        num_queries=100,  num_classes=3)
# model = RFDETRSegMedium(num_queries=100, num_classes=3)
# model = RFDETRSegMedium(num_classes=3)

model_name = model.__class__.__name__
run_dix = datetime.now().strftime("%Y.%m.%d-%H:%M")

model.train(
    dataset_dir="/home/linde/Desktop/Datasets/0619_rfdetr",
    num_workers=16,
    epochs=100, batch_size=2, grad_accum_steps=8,
    lr=3e-3, lr_scheduler='cosine',  # Enables the Cosine Annealing scheduler
    # lr=1e-4,
    resolution=504,
    aug_config=AUG_AGGRESSIVE,
    output_dir=f"output/{model_name}/{run_dix}", progress_bar=False,
    wandb=True,                                  # Enables W&B logging
    project=model_name,                          # Optional: Specify W&B project
    run=run_dix                                  # Optional: Specify W&B run name
)

model.export(opset_version=21, output_dir=f"exported_models/{model_name}/{run_dix}", verbose=False)