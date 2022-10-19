import time
import sys
sys.path.insert( 0, "/volume/huaru/project/signal" )
import logging
import argparse
import os.path as osp
import mmcv
from mmcv import Config
from mmcv.utils import get_logger
from dl.apis import train_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Model')
    parser.add_argument(
        '--config', help='train config file path', default='/volume/huaru/project/signal/configs/SeparationII-moco.py')
    parser.add_argument('--device', help='train device', default='cuda:0')
    return parser.parse_args()


def main():
    # load config file
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # set work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_logger(name=cfg.task_name, log_file=log_file, log_level=logging.INFO)

    #logger.info(f'Config:\n{cfg.pretty_text}')
    #cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # train
    train_model(cfg, logger=logger, device=args.device, meta={})


if __name__ == "__main__":
    main()