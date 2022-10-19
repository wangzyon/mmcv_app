"""
构建信号模拟数据，数据生成配置参考./configs目录下配置文件
"""
import sys
sys.path.insert( 0, "/volume/huaru/project/signal" )
import argparse
import os.path as osp
from mmcv import Config
from mmcv.utils import build_from_cfg
from dl.utils import makedirs
from dl.generators import GENERATOR
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='generate train and val')
    parser.add_argument('--config', help='config file path', default='/volume/huaru/project/signal/configs/SeparationII-moco.py')
    return parser.parse_args()


def main():
    # load config file
    args = parse_args()
    cfg = Config.fromfile(args.config)
    makedirs(osp.join(cfg.train_generator.save_dir, cfg.train_generator.dataset_name), clean=True)
    makedirs(osp.join(cfg.val_generator.save_dir, cfg.val_generator.dataset_name), clean=True)

    train_generator = build_from_cfg(cfg.train_generator, GENERATOR)
    val_generator = build_from_cfg(cfg.val_generator, GENERATOR)

    for _ in tqdm(train_generator, total=len(train_generator), desc=f"generate train signal"):
        pass

    for _ in tqdm(val_generator, total=len(val_generator), desc=f"generate val signal"):
        pass


if __name__ == '__main__':
    main()
