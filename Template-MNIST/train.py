import logging, torch, sys
from utils.load_dataset import get_train_dataset, get_val_dataset
from utils import Trainer
from utils.common import load_config, logging_handler
from model.resnet import MLP

logging.basicConfig(level=logging.INFO, handlers=[logging_handler])


if __name__ == '__main__':
    # 1. Load configurations for the trainer and neural network.
    config = load_config("./config/trainer_params.yaml")
    Trainer.set_hyper_params(config)

    # 2. Prepare dataset.
    dataset_config = config["Dataset"]
    train_dataset = get_train_dataset(dataset_config["TrainDir"], dataset_config["BatchSize"])
    val_dataset = get_val_dataset(dataset_config["ValDir"], dataset_config["BatchSize"])    

    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     logging.info(f'Model loaded from {args.load}')

    # 3. Instantiate the neural network and trainer.
    net = MLP()
    trainer = Trainer(network=net, project_name="MNIST")
    trainer.set_dataset(train_dataset=train_dataset, val_dataset=val_dataset)

    # 4. Train the network.
    try:
        trainer.run(epochs=3)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
