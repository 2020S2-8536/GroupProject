##  18 Oct 2020 Update
Dataloader is now applicable for HICO dataset

<br>


## 13 Oct 2020 Update
Add Pytorch 1.5 support based on old version implementation by chenyuntc.


<br>


## Instructions

## 1. Install dependencies

requires PyTorch >=1.3

- install PyTorch >=1.3 with GPU (code are GPU-only), refer to [official website](http://pytorch.org)

- install cupy, you can install via `pip install cupy-cuda102` or(cupy-cuda110,cupy-cuda101, etc).

- install other dependencies:  `pip install -r requirements.txt`


- start visdom for visualization

```Terminal (in code source root)
python -m visdom.server
```


## 2. Pretrained Model

Download torchvision pretrained model from [Baidu Netdisk](https://pan.baidu.com/s/1UFAT1_I7m1BFfsgUHuo4qw)(passwd: k7eo)


Move the pretrained model to `home/username/.cache/torch/checkpoints/` (OS: Ubuntu 20.04)


## 3. Train

### 3.1 Prepare data

#### Pascal VOC2007

1. Download the dataset from BaiduNet Disk: [Download Link](https://pan.baidu.com/s/1DSGJ3NRnm9lGcY-3woQNIQ) pwd:4n2o.


2. Extract the VOC folder to code source root


3. modify `voc_data_dir` cfg item in `utils/config.py`, or pass it to program using argument like `--voc-data-dir=../VOC/VOCdevkit/VOC2007/` .

### 3.2 begin training

```Bash
mkdir checkpoints/ # folder for snapshots
```

```bash
python train.py train --env='fasterrcnn-caffe' --plot-every=100
```

you may refer to `utils/config.py` for more argument.

Some Key arguments:

- `--caffe-pretrain=False`: use pretrain model from caffe or torchvision (Default: torchvison). ** Note that this version does not support Caffe pretrained Model**
- `--plot-every=n`: visualize prediction, loss etc every `n` batches.
- `--env`: visdom env for visualization
- `--voc_data_dir`: where the VOC data stored
- `--use-drop`: use dropout in RoI head, default False
- `--use-Adam`: use Adam instead of SGD, default SGD. (You need set a very low `lr` for Adam)
- `--load-path`: pretrained model path, default `None`, if it's specified, it would be loaded.

you may open browser, visit `http://<ip>:8097` and see the visualization of training procedure as below:


