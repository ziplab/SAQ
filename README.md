# Sharpness-aware Quantization for Deep Neural Networks

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**This is the official repository for our paper:** [Sharpness-aware Quantization for Deep Neural Networks](https://arxiv.org/abs/2111.12273) by [Jing Liu](https://www.jing-liu.com/), [Jianfei Cai](https://jianfei-cai.github.io/), and [Bohan Zhuang](https://bohanzhuang.github.io/).

## Recent Update

**`2021.11.24`**: We release the source code of SAQ.

## Setup the environments

1. Clone the repository locally:

```bash
git clone https://github.com/zhuang-group/SAQ
```

2. Install pytorch 1.8+, tensorboard and prettytable

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tensorboard
pip install prettytable
```

## Data preparation

### ImageNet

1. Download the ImageNet 2012 dataset from [here](http://image-net.org/), and prepare the dataset based on this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4). 

2. Change the dataset path in `link_imagenet.py` and link the ImageNet-100 by
```bash
python link_imagenet.py
```

### CIFAR-100

Download the CIFAR-100 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz).

After downloading ImageNet and CIFAR-100, the file structure should look like:

```
dataset
├── imagenet
    ├── train
    │   ├── class1
    │   │   ├── img1.jpeg
    │   │   ├── img2.jpeg
    │   │   └── ...
    │   ├── class2
    │   │   ├── img3.jpeg
    │   │   └── ...
    │   └── ...
    └── val
        ├── class1
        │   ├── img4.jpeg
        │   ├── img5.jpeg
        │   └── ...
        ├── class2
        │   ├── img6.jpeg
        │   └── ...
        └── ...
├── cifar100
    ├── cifar-100-python
    │   ├── meta
    │   ├── test
    │   ├── train
    │   └── ...
    └── ...
```


## Training

### Fixed-precision quantization

1. Download the pre-trained full-precision models from the [model zoo](https://github.com/zhuang-group/SAQ/wiki/Model-Zoo).
   
2. Train low-precision models.

To train low-precision ResNet-20 on CIFAR-100, run:

```bash
sh script/train_qsam_cifar_r20.sh
```

To train low-precision ResNet-18 on ImageNet, run:

```bash
sh script/train_qsam_imagenet_r18.sh
```

### Mixed-precision quantization

1. Download the pre-trained full-precision models from the [model zoo](https://github.com/zhuang-group/SAQ/wiki/Model-Zoo).

2. Train the configuration generator.

To train the configuration generator of ResNet-20 on CIFAR-100, run:

```bash
sh script/train_generator_cifar_r20.sh
```

To train the configuration generator on ImageNet, run:

```bash
sh script/train_generator_imagenet_r18.sh
```

3. After training the configuration generator, run following commands to fine-tune the resulting models with the obtained bitwidth configurations on CIFAR-100 and ImageNet.
```bash
sh script/finetune_cifar_r20.sh
```

```bash
sh script/finetune_imagenet_r18.sh
```

## Results on CIFAR-100

| Network | Method | Bitwidth | BOPs (M) | Top-1 Acc. (%) | Top-5 Acc. (%) |
| :-----: | :----: | :------: | :------: | :------------: | :------------: |
| ResNet-20 | SAQ | 4 | 674.6 | 68.7 | 91.2 |
| ResNet-20 | SAMQ | MP | 659.3 | 68.7 | 91.2 |
| ResNet-20 | SAQ | 3 | 392.1 | 67.7 | 90.8 |
| ResNet-20 | SAMQ | MP | 374.4 | 68.6 | 91.2 |
| MobileNetV2 | SAQ | 4 | 1508.9 | 75.6 | 93.7 |
| MobileNetV2 | SAMQ | MP | 1482.1 | 75.5 | 93.6 |
| MobileNetV2 | SAQ | 3 | 877.1 | 74.4 | 93.2 |
| MobileNetV2 | SAMQ | MP | 869.5 | 75.5 | 93.7 |

## Results on ImageNet

| Network | Method | Bitwidth | BOPs (G) | Top-1 Acc. (%) | Top-5 Acc. (%) |
| :-----: | :----: | :------: | :------: | :------------: | :------------: |
| ResNet-18 | SAQ | 4 | 34.7 | 71.3 | 90.0 |
| ResNet-18 | SAMQ | MP | 33.7 | 71.4 | 89.9 |
| ResNet-18 | SAQ | 2 | 14.4 | 67.1 | 87.3 |
| MobileNetV2 | SAQ | 4 | 5.3 | 70.2 | 89.4 |
| MobileNetV2 | SAMQ | MP | 5.3 | 70.3 | 89.4 |


## Citation
If you find *SAQ* useful in your research, please consider to cite the following related papers:
```
@article{liu2021sharpnessaware,
    title={Sharpness-aware Quantization for Deep Neural Networks}, 
    author={Jing Liu and Jianfei Cai and Bohan Zhuang},
    journal={arXiv preprint arXiv:2111.12273},
    year={2021},
}
```

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Acknowledgement

This repository has adopted codes from [SAM](https://github.com/davda54/sam), [ASAM](https://github.com/SamsungLabs/ASAM) and [ESAM](https://github.com/dydjw9/efficient_sam), we thank the authors for their open-sourced code.
