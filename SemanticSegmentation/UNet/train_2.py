import argparse, logging, torch, sys
from model import UNet
from utils import Trainer



if __name__ == '__main__':
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)

    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     logging.info(f'Model loaded from {args.load}')

    trainer = Trainer(network=net)

    try:
        trainer.run(epochs=50)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
