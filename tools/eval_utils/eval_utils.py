import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models.dense_heads import AnchorFreeSingle
from pcdet.models.detectors import CenterPoint
from pcdet.models.backbones_2d.map_to_bev import PointPillarScatter
from pcdet.models.backbones_3d.vfe import PillarVFE
def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

count = 0
def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model[0].eval()
    model[1].eval()
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        # print('data123', data)
        # if i == 200:
        #     break
        Vfe = PillarVFE(model_cfg=cfg.MODEL, num_point_features=4, voxel_size=[0.16, 0.16, 4],
                      point_cloud_range=dataloader.dataset.point_cloud_range)
        voxel_size = [0.16, 0.16, 4]
        point_cloud_range = dataloader.dataset.point_cloud_range
        x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        z_offset = voxel_size[2] / 2 + point_cloud_range[2]
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']
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
        batch_dict['voxel_num_points'] = voxel_num_points
        batch_dict['voxel_coords'] = coords
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / (voxel_num_points.type_as(voxel_features).view(
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
        features[cur_num:, :, :] = 0.0
        features = features.permute(2, 0, 1)
        features = features.unsqueeze(dim=0)
        # import pdb
        # pdb.set_trace()
        # datalist1.append(features.detach().cpu())
        # np.save("./bin/"+str(i)+".npy",data['img'])
        # import pdb;
        # pdb.set_trace()
        # datalist.append((list(torch.from_numpy(data['bev_map'])), 1))
    # for i, batch_dict in enumerate(dataloader): # 0~3769
    #     load_data_to_gpu(batch_dict)
        # datalist.append(batch_dict['bev_map'])

        AnchorFreeHead = AnchorFreeSingle(model_cfg=cfg.MODEL, input_channels=64, num_class=len(cfg.CLASS_NAMES),
                             class_names=cfg.CLASS_NAMES, grid_size=dataloader.dataset.grid_size,
                             point_cloud_range=dataloader.dataset.point_cloud_range)
        Detector = CenterPoint(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataloader.dataset)
        Maptobev = PointPillarScatter(model_cfg=cfg.MODEL, grid_size=dataloader.dataset.grid_size)
        with torch.no_grad():

            # pred_dicts, ret_dict = model(batch_dict)

            features = model[0]([features])
            # import pdb
            # pdb.set_trace()
            features = features.squeeze(dim=0)
            features = features.permute(1, 2, 0)
            features = features.squeeze()
            # features = features.squeeze()
            # import pdb
            # pdb.set_trace()
            batch_dict['pillar_features'] = features
            batch_dict =  Maptobev(batch_dict)
            # global count
            #
            # import skimage
            # jpg = batch_dict['spatial_features'].detach().cpu()[0].abs().sum(0).numpy()
            # print('-----------------------', jpg.min(), jpg.max())
            # jpg = ((jpg - jpg.min()) / (jpg.max() - jpg.min()) * 255 ).astype(np.ubyte)
            # # jpg = jpg.transpose(1,2,0)
            # skimage.io.imsave('./bev_%d.jpg' % count, jpg)
            # count += 1
            # import pdb
            # pdb.set_trace()
            # cls_preds,box_preds = model[1]([batch_dict['bev_map']])
            # spatial_features = batch_dict['spatial_features']

            # spatial_features = np.load('./feat.npy')
            # spatial_features = torch.from_numpy(spatial_features).cuda()
            # batch_dict['spatial_features'] = spatial_features

            cls_preds, box_preds = model[1]([batch_dict['spatial_features']] )

            # cls_preds = np.load('./cls_preds.npy')
            # cls_preds = torch.from_numpy(cls_preds).cuda()
            # box_preds = np.load('./box_preds.npy')
            # box_preds = torch.from_numpy(box_preds).cuda()

            pred_dicts = AnchorFreeHead.generate_predicted_boxes( batch_dict['batch_size'], cls_preds, box_preds)
            # print(pred_dicts)
            data_dict =batch_dict
            data_dict['final_box_dicts']=pred_dicts
            pred_dicts, ret_dict = Detector.post_processing(data_dict)


        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
