import os
import options
import sys
import torch

from engine import Engine
from misc.utils import log


def main():
    log.process(os.getpid())
    log.title("[{}] (PyTorch code for testing MuRF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd, load_confd=False)
    options.save_options_file(opt)

    opt.dist = False
    opt.local_rank = 0

    with torch.cuda.device(opt.device):
        m = Engine(opt)

        m.build_networks()
        m.restore_checkpoint()

        split = 'val' if getattr(opt, 'test_on_val_set', False) else 'test'

        m.load_dataset(splits=[split])

        if opt.nerf.render_video:
            m.test_model_video(leave_tqdm=True,
            save_depth_video=getattr(opt, 'save_depth_video', False),
            save_depth_np=getattr(opt, 'save_depth_np', False),
            )
        else:
            m.test_model(leave_tqdm=True, save_depth=getattr(opt, 'save_depth', False),
                         save_gt_depth=getattr(opt, 'save_gt_depth', False),
                         with_depth_metric=getattr(opt, 'with_depth_metric', False),
                         save_images=getattr(opt, 'save_imgs', False),
                         save_depth_np=getattr(opt, 'save_depth_np', False),
                         save_gt_depth_np=getattr(opt, 'save_gt_depth_np', False),
                         )


if __name__ == "__main__":
    main()
