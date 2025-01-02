# Class-aware Multi-granularity Co-Diffusion Models for Learning with Noisy labels on Imbalanced Datasets

## 1. Preparing python environment

Install requirements.<br />

```
conda env create -f environment.yml
conda activate CaMCoD-env
conda env update --file environment.yml --prune
```

The name of the environment is set to **CaMCoD-env** by default. You can modify the first line of the `environment.yml` file to set the new environment's name.

## 2. Pre-trained model & Checkpoints

* CLIP models are available in the python package at [here](https://github.com/openai/CLIP). Install without dependency: <br />

```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git  --no-dependencies
```

Trained checkpoints for the diffusion models are available at [here]().

## 3. Run demo script to train the CaMCoD models

### 3.1 Balanced CIFAR-10 and CIFAR-100<br />

Default values for input arguments are given in the code. An example command is given:

```
python train_on_CIFAR.py --device cuda:0 --noise_type cifar10-sym-0.4 --nepoch 200 --warmup_epochs 5 --log_name cifar10_sym_0.4.log
```

### 3.2 Imbalanced CIFAR-10 and CIFAR-100<br />

Default values for input arguments are given in the code. An example command is given:

```
python train_on_im_CIFAR.py --device cuda:0 --noise_type cifar10-sym-0.4 --imbalanced_rho 10 --nepoch 200 --warmup_epochs 5 --log_name im_cifar10_sym_0.4.log
```

### 3.3 Animal10N<br />
The dataset should be downloaded according to the instruction here: [Aniaml10N](https://dm.kaist.ac.kr/datasets/animal-10n/)<br />
Default values for input arguments are given in the code. An example command is given:
```
python train_on_Animal10N.py --device cuda:0 --nepoch 200 --warmup_epochs 5 --log_name Animal10N.log
```

### 3.4 WebVision and ILSVRC2012<br />
Download [WebVision 1.0](https://data.vision.ee.ethz.ch/cvl/webvision/download.html) and the validation set of [ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/) datasets. The ImageNet synsets labels for ILSVRC2012 validation set is provided [here](https://data.vision.ee.ethz.ch/cvl/webvision/).
```

python train_on_WebVision.py --gpu_devices 0 1 2 3 4 5 6 7 --nepoch 200 --warmup_epochs 5  --log_name Webvision.log

python test_on_ILSVRC2012.py --gpu_devices 0 1 2 3 4 5 6 7 --log_name ILSVRC2012.log

```

### 3.5 Clothing1M<br />
The dataset should be downloaded according to the instruction here: [Clothing1M](). Default values for input arguments are given in the code. <br />

```

python train_Clothing1M.py --gpu_devices 0 1 2 3 4 5 6 7 --nepoch 200 --warmup_epochs 5  --log_name Clothing1M.log

```
