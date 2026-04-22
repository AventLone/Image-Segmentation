from datetime import datetime
from rfdetr import RFDETRSegMedium

model = RFDETRSegMedium(num_queries=100, num_select=80, num_classes=2)
model.train(
    dataset_dir="/home/avent/Desktop/generated_data/rf",
    epochs=36, batch_size=4, grad_accum_steps=4, 
    lr=1e-4, lr_scheduler='cosine',  # Enables the Cosine Annealing scheduler
    output_dir="output", progress_bar=False,
    wandb=True,                                         # Enables W&B logging
    project="Instance Segmentation",                    # Optional: Specify W&B project
    run=datetime.now().strftime("%Y-%m-%d-%H%M%S")      # Optional: Specify W&B run name
)

model.export(output_dir="exported_models")

# model.export(
#     format="tensorrt",      # Exports as a TensorRT .engine/.plan file
#     output_dir="tensorrt_weights",
#     half=True,              # Export in FP16 for significant speedup
#     simplify=True           # Cleans the graph for better compatibility
# )