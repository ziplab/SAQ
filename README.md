# Sharpness-aware Quantization for Deep Neural Networks

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Recent Update

**`2021.11.23`**: We release the source code of SAQ.

## Setup the environments

1. Clone the repository locally:

```
git clone https://github.com/zhuang-group/SAQ
```

2. Install pytorch 1.8+, tensorboard and prettytable

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tensorboard
pip install prettytable
```

## Data preparation

### ImageNet

Download the ImageNet 2012 dataset from [here](http://image-net.org/), and prepare the dataset based on this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4). 

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

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Acknowledgement

This repository has adopted codes from [SAM](https://github.com/davda54/sam), [ASAM](https://github.com/SamsungLabs/ASAM) and [ESAM](https://github.com/dydjw9/efficient_sam), we thank the authors for their open-sourced code.