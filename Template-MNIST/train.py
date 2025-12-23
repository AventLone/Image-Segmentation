import logging, torch, sys
from utils.load_dataset import get_train_dataset, get_val_dataset, get_datasets
from utils import Trainer
from utils.common import load_config, logging_handler
from model.resnet import MLP, SmallCNN, ResNet

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = False   # Set True for full determinism (slower)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")   # Allow TF32 on Ampere+ for speed

    logging.basicConfig(level=logging.INFO, handlers=[logging_handler])

    # 1. Load configurations for the trainer and neural network.
    config = load_config("./config/trainer_params.yaml")
    Trainer.set_hyper_params(config)

    # 2. Prepare dataset.
    dataset_config = config["Dataset"]
    train_loader, val_loader = get_datasets(dir_path=dataset_config["TrainDir"], 
                                            batch_size=dataset_config["BatchSize"], 
                                            val_ratio=dataset_config["ValidationRatio"])

    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     logging.info(f'Model loaded from {args.load}')

    # 3. Instantiate the neural network and trainer.
    net = ResNet()
    trainer = Trainer(network=net, project_name="MNIST")
    trainer.set_dataset(train_dataset=train_loader, val_dataset=val_loader)

    # 4. Train the network.
    try:
        trainer.run(epochs=6)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
