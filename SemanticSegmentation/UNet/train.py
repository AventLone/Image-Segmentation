import logging, torch, sys
from model import UNet
from utils import Trainer
from utils.common import load_config, logging_handler

logging.basicConfig(level=logging.INFO, handlers=[logging_handler])


if __name__ == '__main__':

    # 1. Prepare datasets


    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    
    config = load_config("./config/trainer_params.yaml")
    Trainer.set_hyper_params(config["Trainer"])

    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     logging.info(f'Model loaded from {args.load}')

    net = UNet(config["Network"])
    trainer = Trainer(network=net)
    try:
        trainer.run(epochs=50)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
