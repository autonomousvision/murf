import torch
import sys
import os

from engine import Engine
from misc.utils import log
import options

import argparse

from misc.dist_utils import get_dist_info, init_dist, setup_for_distributed


def main():
    log.process(os.getpid())
    log.title("[{}] (PyTorch code for training MuRF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd, load_confd=True)
    options.save_options_file(opt)

    # distributed training
    if getattr(opt, 'dist', False):
        print('distributed training')
        dist_params = dict(backend='nccl')
        launcher = getattr(opt, 'launcher', 'pytorch')
        init_dist(launcher, **dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        opt.gpu_ids = range(world_size)
        opt.local_rank = int(os.environ['LOCAL_RANK'])
        opt.device = torch.device('cuda:{}'.format(opt.local_rank))

        setup_for_distributed(opt.local_rank == 0)

    else:
        opt.local_rank = 0
        opt.dist = False

    m = Engine(opt)

    # setup model
    m.build_networks()

    # setup dataset
    if getattr(opt, 'no_val', False):
        m.load_dataset(splits=['train', 'test'])
    else:
        m.load_dataset(splits=['train', 'val', 'test'])

    # setup trianing utils
    m.setup_visualizer()
    m.setup_optimizer()

    if opt.resume or opt.load:
        m.restore_checkpoint()

    m.train_model()


if __name__=="__main__":
    main()
