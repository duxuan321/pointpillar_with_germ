import _init_path
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
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=2, required=False, help='batch size for training')
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
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
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


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir, save_to_file=args.save_to_file
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)

ccc = 0
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
    datalist1 = []
    datalist2 = []
    for i, batch_dict in enumerate(test_loader):

        if i == 200:
            break

        # np.save("./bin/"+str(i)+".npy",data['img'])
        # import pdb;
        # pdb.set_trace()
        load_data_to_gpu(batch_dict)
        _, outputs = model(batch_dict)
        datalist2.append(outputs['spatial_features'].detach().cpu())
        # print(outputs['spatial_features'])
        Vfe = PillarVFE(model_cfg=cfg.MODEL,num_point_features=4, voxel_size=[0.16,0.16,4], point_cloud_range=test_loader.dataset.point_cloud_range)
        voxel_size = [0.16,0.16,4]
        point_cloud_range = test_loader.dataset.point_cloud_range
        x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        z_offset = voxel_size[2] / 2 + point_cloud_range[2]
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # print(voxel_features.max(),voxel_features.min())
        # print(voxel_features.shape, voxel_features[:, :, :3].shape)

        maxnum = 8694
        cur_num = voxel_features.shape[0]
        assert maxnum >= cur_num
        temp = torch.zeros([maxnum - voxel_features.shape[0], voxel_features.shape[1], voxel_features.shape[2]]).cuda()
        voxel_features = torch.cat((voxel_features, temp), dim=0)
        temp = torch.zeros([maxnum - voxel_num_points.shape[0]]).cuda()
        voxel_num_points = torch.cat((voxel_num_points, temp), dim=0)
        temp = torch.zeros([maxnum - coords.shape[0], coords.shape[1]]).cuda()
        coords = torch.cat((coords, temp), dim=0)
        batch_dict['voxels'] = voxel_features
        # print(voxel_features.max(), voxel_features.min())
        # print(voxel_features.shape, voxel_features[:, :, :3].shape)
        batch_dict['voxel_num_points'] = voxel_num_points
        batch_dict['voxel_coords'] = coords
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / ( voxel_num_points.type_as(voxel_features).view(
            -1, 1, 1) + 0.00001)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                    coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * voxel_size[0] + x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                    coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * voxel_size[1] + y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                    coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * voxel_size[2] + z_offset)

        features = [voxel_features, f_cluster, f_center]
        features = torch.cat(features, dim=-1)
        voxel_count = features.shape[1]
        mask = Vfe.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        weight = torch.Tensor(
            [1 / 70, 1 / 80, 1 / 4, 1.0, 1 / 0.32, 1 / 0.32, 1 / 8, 1 / 0.16, 1 / 0.16, 1 / 4]).cuda().view(1, 1, -1)
        bias = torch.Tensor([-0.5, 0.0, 0.25, -0.5, 0, 0, 0, 0, 0, 0]).cuda().view(1, 1, -1)
        features = features * weight + bias
        features *= mask
        features[cur_num:, :, :] = 0.0 #features[0:1, :, :].contiguous()
        features = features.permute(2, 0, 1)
        features = features.unsqueeze(dim=0)
        # import pdb
        # pdb.set_trace()
        # _, outputs = model(batch_dict)
        # import pdb
        # pdb.set_trace()
        datalist1.append(features.detach())

        # import pdb
        # # pdb.set_trace()
        # global ccc
        # if ccc == 3:
        #     np.save('./feat.npy', outputs['spatial_features'].detach().cpu().numpy())
        # ccc += 1
    input_vars1 = [torch.randn((1,10,maxnum,32))]
    work_dir = '/home/yuanxin/mvlidarnet_pcdet'
    onnx_model1 = onnx.load('/home/yuanxin/mvlidarnet_pcdet/pfn.onnx')
    quant_model1 = run_quantizer(onnx_model1, dataloader=datalist1, num_batches=200,
                                output_dir=work_dir + '/ir_output_pfn', input_vars=input_vars1)
    # features = quant_model1(datalist1 )
    # features = features.squeeze(dim=0)
    # features = features.permute(1, 2, 0)
    # features = features.squeeze()
    #
    # Maptobev = PointPillarScatter(model_cfg=cfg.MODEL, grid_size=test_loader.dataset.grid_size)
    # datalist2=[]
    #
    # batch_dict['pillar_features'] = features
    # batch_dict = Maptobev(batch_dict)
    # datalist2.append(batch_dict['spatial_features'].detach())
    input_vars2 = [torch.randn((1, 64, 432, 496))]
    work_dir = '/home/yuanxin/mvlidarnet_pcdet'
    onnx_model2 = onnx.load(
        '/home/yuanxin/mvlidarnet_pcdet/weights/pointpillar_epoch_50_default.onnx')
    quant_model2 = run_quantizer(onnx_model2, dataloader=datalist2, num_batches=200,
                                output_dir=work_dir + '/ir_output_backbone', input_vars=input_vars2)
    # import pdb
    # pdb.set_trace()
    run_compiler(input_dir=work_dir + '/ir_output_pfn', output_dir=work_dir + '/compiler_output_pfn',
                 enable_cmodel=True, enable_rtl_model=True, enable_profiler=False)
    run_compiler(input_dir=work_dir + '/ir_output_backbone', output_dir=work_dir + '/compiler_output_backbone',
                 enable_cmodel=True, enable_rtl_model=True, enable_profiler=False)
    # for i, batch_dict in enumerate(dataloader):
    #     onnx_model(batch_dict)
    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            # eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)
            eval_single_ckpt1([quant_model1,quant_model2], test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)
            # model.vfe.fake_quant = True
            # eval_single_ckpt1([model.vfe, quant_model2], test_loader, args, eval_output_dir, logger, epoch_id,
            #                   dist_test=dist_test)
    import pdb
    pdb.set_trace()
if __name__ == '__main__':
    main()
