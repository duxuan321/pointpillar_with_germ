## 0.安装

pip install -r requirment.txt
python setup.py develop
ln -s /your/data/file ./pcdet/data/kitti

## 1.训练
CUDA_VISIBLE_DEVICES=0,1 bash scripts/dist_train.sh 2 --batch_size 16 --extra_tag with_hei_filter --cfg_file ./cfgs/kitti_models/mvlidarnet.yaml 


## 2.测试
CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file ./cfgs/kitti_models/mvlidarnet.yaml  --batch_size 1 --ckpt ../output/cfgs/kitti_models/mvlidarnet/default/ckpt/checkpoint_epoch_80.pth

## 3.更新
2022/7/7 v1.1

修改了一些BUG，对DBSCAN后处理进行了一些格式化处理，修改了高度过滤


2022/9/27 v2.0

加入了mvlidarnet_center.yaml配置文件，更换了mvldiarnet的head，精度更高，可以接近pointpillar的精度（需要在数据集里加入地面信息）