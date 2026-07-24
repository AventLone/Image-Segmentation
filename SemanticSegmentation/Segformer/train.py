from transformers import SegformerForSemanticSegmentation
from utils.config import TrainConfig
from utils.data import get_dataloaders
from utils.trainer import Trainer
from utils.training_utils import set_seed


def main() -> None:
    # 1. Load configurations.
    configs = TrainConfig()
    set_seed(configs.seed)

    # 2. Prepare dataset.
    train_loader, val_loader, dataset_meta = get_dataloaders(configs)

    # 3. Instantiate the network and trainer.
    net = SegformerForSemanticSegmentation.from_pretrained(configs.pretrained_path,
                                                           num_labels=dataset_meta["num_labels"],
                                                           id2label=dataset_meta["id2label"],
                                                           label2id=dataset_meta["label2id"],
                                                           ignore_mismatched_sizes=True)
    trainer = Trainer(network=net, configs=configs,
                      num_labels=dataset_meta["num_labels"],
                      id2label=dataset_meta["id2label"],
                      label2id=dataset_meta["label2id"],
                      project_name="Semantic Segmentation")
    trainer.set_dataset(train_dataset=train_loader, val_dataset=val_loader,
                        train_size=dataset_meta["train_size"],
                        val_size=dataset_meta["val_size"])

    # 4. Train.
    trainer.run(configs.epochs)


if __name__ == "__main__":
    main()
