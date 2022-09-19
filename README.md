## install
python setup.py develop

//在pointpillar.yaml 的DATA_PATH中修改你KITTI数据集的位置

## train
python -m torch.distributed.launch --nproc_per_node=4 train.py --launcher pytorch --cfg_file ./cfgs/kitti_models/pointpillar.yaml

## float test
python test.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml  --ckpt ../weights/checkpoint_epoch_20.pth

## pointpillar分段式导出onnx
## 导出onnx_split
python export_onnx_split.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml --ckpt ../weights/checkpoint_epoch_20.pth

## quant & test
python quant_split.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml --ckpt ../weights/checkpoint_epoch_20.pth

## pointpillar一次性导出onnx
note 
量化工具nquantizer中graph.py line 70
if not isinstance(input_tensors, list):改为if not isinstance(input_tensors, (list, tuple)):
## 导出onnx
python export_onnx.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml --ckpt ../weights/checkpoint_epoch_20.pth

## quant & test
python quant.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml --ckpt ../weights/checkpoint_epoch_20.pth


## 测试结果记录
https://lekqjyg0qj.feishu.cn/docx/doxcnoGr48VgV9szuncjEleGdXd

https://lekqjyg0qj.feishu.cn/sheets/shtcntkFN8PoK3K2w4voN4s79ah


## 量化编译工具安装


修改代码时所用工具链版本：pytorch ==1.8.0  , cuda = 10.2

工具链文档:http://192.168.3.224:8090/pages/viewpage.action?pageId=33555282&navigatingVersions=true

安装：

git clone --recurse-submodules http://192.168.3.224:8081/toolchain/npu_quantizer

git clone http://192.168.3.224:8081/toolchain/npu_compiler

cd npu_quantizer && pip install -e n-graph && pip install -r requirements.txt && pip install -e .

cd npu_compiler && pip install -r requirements.txt && pip install -e .

初次运行量化编译工具事需要密钥key,@工具链郑成林

更详细的《超星未来NPU工具链开发手册》参见http://192.168.3.224:8081/toolchain/npuv1_tool_docs/tree/master
