import sys, logging, torch
from utils.load_dataset import get_dataloaders
from utils import Trainer
from utils.common import load_config, logging_handler
from model import UNet

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = False   # Set True for full determinism (slower)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")   # Allow TF32 on Ampere+ for speed

    logging.basicConfig(level=logging.INFO, handlers=[logging_handler])

    # 1. Load configurations for the trainer and neural network.
    config = load_config("./config/trainer_params.yaml")
    Trainer.set_hyper_params(config)

    logging.info(
            f"Trainer Parameters:\n"
            f"\t Batch size:      {Trainer.batch_size}\n"
            f"\t Learning rate:   {Trainer.learning_rate}\n"
            f"\t Checkpoints:     {Trainer.save_checkpoint}\n"
            f"\t Device:          {Trainer.device.type}\n"
            f"\t Image scaling:   {Trainer.img_scale}\n"
            f"\t Mixed Precision: {Trainer.amp}"
        )

    # 2. Prepare dataset.
    dataset_config = config["Dataset"]
    train_loader, val_loader = get_dataloaders(dir_path=dataset_config["TrainDir"],
                                               batch_size=dataset_config["BatchSize"],
                                               val_ratio=dataset_config["ValidationRatio"])

    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     logging.info(f'Model loaded from {args.load}')

    # 3. Instantiate the neural network and trainer.
    net = UNet(channels_num=3, classes_num=5)
    trainer = Trainer(network=net)
    trainer.set_dataset(train_dataset=train_loader, val_dataset=val_loader)

    # 4. Train the network.
    try:
        trainer.run(epochs=100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
