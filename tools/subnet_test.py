
import argparse
import datetime
import glob
from operator import mod
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
import yaml
from thop import profile
from icecream import ic 


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--sample_num', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    
    #! load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()
    ic(epoch_id)

    yaml_file_path = str(args.sample_num) + "_sample.yaml"
    json_file_path = str(args.sample_num) + "_sample.json"
    # with open(yaml_file_path, "x") as of:
    #     for arch_id in range(args.sample_num):
    #         of.write("# ---- Arch {} ----\n".format(arch_id))
    #         yaml.safe_dump([model.backbone_2d.search_space.random_sample().genotype], of)
    #         of.write("\n")
    assert os.path.exists(yaml_file_path)
    with open(yaml_file_path, "r") as fr:
        samples = yaml.load(fr)
    
    AP_car = {}
    AP_pes = {}
    AP_cyc = {}
    AP_total = {}
    tested_geno = set()
    with torch.no_grad():
        for arch_id, geno in enumerate(samples):
            ic(arch_id)
            if geno in tested_geno: #避免重复架构的测试
                continue
            model.gene_type = geno

            
            # for i, batch_dict in enumerate(test_loader):
            #     assert batch_dict['batch_size'] == 1
            #     flops, params = profile(model, inputs=(batch_dict,))
            #     print(flops/1e9,params/1e6) #flops单位G，para单位M
            #     ic(flops/1e9, params/1e6)
            # break



            # start evaluation
            result_dict = eval_utils.eval_one_epoch(
                    cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
                    result_dir=eval_output_dir, save_to_file=args.save_to_file, arch_id=arch_id)
                # import pdb;pdb.set_trace()
            if cfg.LOCAL_RANK == 0: #!    
                AP_car[arch_id] = result_dict['Car_3d/moderate_R40']
                AP_pes[arch_id] = result_dict['Pedestrian_3d/moderate_R40']
                AP_cyc[arch_id] = result_dict['Cyclist_3d/moderate_R40']
                AP_total[arch_id] = AP_car[arch_id] + AP_pes[arch_id] + AP_cyc[arch_id]
            tested_geno.add(geno)
    ic(AP_car)
    import json
    if cfg.LOCAL_RANK == 0: #!
        with open(json_file_path, "a") as fw:
            sorted_AP_car = {k:v for k,v in sorted(AP_car.items(), key = lambda kv:kv[1])[::-1]}
            sorted_AP_pes = {k:v for k,v in sorted(AP_pes.items(), key = lambda kv:kv[1])[::-1]}
            sorted_AP_cyc = {k:v for k,v in sorted(AP_cyc.items(), key = lambda kv:kv[1])[::-1]}
            sorted_AP_total = {k:v for k,v in sorted(AP_total.items(), key = lambda kv:kv[1])[::-1]}
            json.dump(sorted_AP_car, fw)
            json.dump(sorted_AP_pes, fw)
            json.dump(sorted_AP_cyc, fw)
            json.dump(sorted_AP_total, fw)

if __name__ == '__main__':
    main()
