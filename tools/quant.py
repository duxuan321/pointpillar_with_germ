
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
import onnx
import numpy as np
# print(np.__version__)
import torch
from tensorboardX import SummaryWriter
from pcdet.models.backbones_2d.map_to_bev import PointPillarScatter
from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from nquantizer import run_quantizer
from ncompiler import run_compiler
from pcdet.models import load_data_to_gpu
from pcdet.models.backbones_3d.vfe import PillarVFE
from torch import nn
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
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
    parser.add_argument('--save_to_file', action='store_true', default=True, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    model.cuda()
    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )
def eval_single_ckpt1(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )

def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None

class QuantModel(nn.Module):
    def __init__(self, src_model, part1, part2=None):
        super().__init__()
        self.model = src_model
        self.part1 = part1
        self.part2 = part2

    def forward(self, batch_dict):
        voxels = batch_dict['voxels']
        pillar_features = self.part1([voxels, ])
        batch_dict.update({'pillar_features': pillar_features})
        batch_dict = self.model.map_to_bev_module(batch_dict)
        spatial_features = batch_dict['spatial_features']
        if self.part2 is None:
            return spatial_features
        cls_preds, box_preds, dir_cls_preds = self.part2([spatial_features, ])
        batch_dict.update({'cls_preds': cls_preds, 'box_preds': box_preds, 'dir_cls_preds': dir_cls_preds})
        batch_dict = self.model.dense_head.forward_quant(batch_dict)

        pred_dicts, recall_dicts = self.model.post_processing(batch_dict)
        return pred_dicts, recall_dicts

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

    output_dir = cfg.ROOT_DIR / 'output'  / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
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
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)

    model.cuda()
    model.eval()

    work_dir = '../quant_results_benewake_pillar'
    os.makedirs(work_dir, exist_ok=True)
    onnx_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag

    onnx_part1_path = os.path.join(onnx_dir, 'deploy_part1.onnx')
    onnx_part2_path = os.path.join(onnx_dir, 'deploy_part2.onnx')


    datalist1 = []
    for i, batch_dict in enumerate(test_loader):
        if i == 200:
            break
        datalist1.append(torch.from_numpy(batch_dict['voxels']).float())

    onnx_model1 = onnx.load(onnx_part1_path)
    quant_model1 = run_quantizer(onnx_model1, dataloader=datalist1, num_batches=200,
                                output_dir=work_dir + '/ir_output_pfn', input_vars=datalist1[0], enable_equalization=True)

    # run_compiler(input_dir=work_dir + '/ir_output_pfn', output_dir=work_dir + '/compiler_output_pfn',
    #              enable_cmodel=True, enable_rtl_model=True, enable_profiler=True)

    datalist2 = []
    quant_model = QuantModel(model, quant_model1)
    quant_model.eval()
    quant_model.cuda()
    for i, batch_dict in enumerate(test_loader):
        if i == 200:
            break
        load_data_to_gpu(batch_dict)
        spatial_features = quant_model(batch_dict)
        datalist2.append(spatial_features.detach().cpu())

    onnx_model2 = onnx.load(onnx_part2_path)
    quant_model2 = run_quantizer(onnx_model2, dataloader=datalist2, num_batches=200,
                                output_dir=work_dir + '/ir_output_backbone2d', input_vars=datalist2[0], enable_equalization=True)

    # run_compiler(input_dir=work_dir + '/ir_output_backbone2d', output_dir=work_dir + '/compiler_output_backbone2d',
    #              enable_cmodel=True, enable_rtl_model=True, enable_profiler=True)

    quant_model = QuantModel(model, quant_model1, quant_model2)
    with torch.no_grad():
        eval_single_ckpt(quant_model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)




if __name__ == '__main__':
    main()
