import sys, logging, torch
from utils.load_dataset import get_dataloaders
from utils import Trainer, TrainConfigs
from utils.common import logging_handler
from model import UNet

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, handlers=[logging_handler])

    # Get "epochs" from command
    if len(sys.argv) != 2:
        logging.fatal("Wrong format for the training command!")
        sys.exit(0)
        
    epochs = int(sys.argv[1])

    if epochs == 0 or epochs is None:
        logging.fatal("epochs must bigger than 0!")
        sys.exit(0)

    torch.backends.cudnn.deterministic = False   # Set True for full determinism (slower)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")   # Allow TF32 on Ampere+ for speed

    
    # 1. Load configurations for the trainer and neural network.
    config = TrainConfigs("./config/trainer_params.yaml")
    config.print_info()

    # 2. Prepare dataset.
    train_loader, val_loader = get_dataloaders(configs=config)

    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     logging.info(f'Model loaded from {args.load}')

    # 3. Instantiate the neural network and trainer.
    net = UNet(channels_num=config.input_size[0], classes_num=config.classes_num)
    trainer = Trainer(network=net, configs=config, project_name="Semantic Segmentation")
    trainer.set_dataset(train_dataset=train_loader, val_dataset=val_loader)

    # 4. Train the network.
    try:
        trainer.run(epochs)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.warning('User interrupted, saved the trained parameters file.')
        sys.exit(0)