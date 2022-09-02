## 0.install
python setup.py develop

//在pointpillar.yaml 的DATA_PATH中修改你KITTI数据集的位置

## 1.train
CUDA_VISIBLE_DEVICES=2,3 bash scripts/dist_train.sh 2 --batch_size 8 --extra_tag  test --cfg_file ./cfgs/kitti_models/pointpillar.yaml

## 2.test
#quant
#pointpillar
CUDA_VISIBLE_DEVICES=3 python test_part_pfn.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml --batch_size 1 --ckpt /home/yuanxin/mvlidarnet_pcdet/weights/checkpoint_epoch_50.pth
# float
#pointpillar
CUDA_VISIBLE_DEVICES=1 python test_origin.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml --batch_size 1  --ckpt /home/yuanxin/mvlidarnet_pcdet/weights/checkpoint_epoch_50.pth
## 3.导出onnx
// 将pointpillar.yaml配置文件中的两处EXPORT_ONNX 设为True（训练测试的时候要改回来）
# backbone以及head
python export_onnx.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml --batch_size 4 --ckpt /home/yuanxin/mvlidarnet_pcdet/weights/checkpoint_epoch_50.pth
#vfe中pnl部分
//将pillar_vfe.py文件中的export_onnx设为True(测试训练时要改回来)
CUDA_VISIBLE_DEVICES=3 python test_origin.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml --batch_size 1  --ckpt /home/yuanxin/mvlidarnet_pcdet/weights/checkpoint_epoch_50.pth
## 4.导出jitscript格式的模型
python export_jitscript.py --cfg_file ./cfgs/kitti_models/mvlidarnet.yaml --batch_size 4 --ckpt ../output/epoch_80.pth