## StyleWSGAN

This repository contains code to reproduce experiments presented in:


**Generative Modeling Helps Weak Supervision (and Vice Versa)**<br>
Benedikt Boecking, Nicholas Roberts, Willie Neiswanger, Stefano Ermon, Frederic Sala, Artur Dubrawski<br>
International Conference on Learning Representations (ICLR) (2023)<br>
<a class="" href="https://arxiv.org/abs/2203.12023">[arXiv]</a> <a class="" href="https://openreview.net/forum?id=3OaBBATwsvP">[OpenReview]</a>

With the code in this repository, a weakly supervised GAN (WSGAN) can be trained on weakly supervised images, using a StyleGAN2 model as its base network. The WSGAN, which consists of a GAN and a label model, is trained on image datasets where some of the images have one or more labeling function (LF) votes associated that output weak labels, following the programmatic weak supervision/data programming paradigm. The code for the main experiments presented in the paper above, where a simple DCGAN is used as the base network, can be found at https://github.com/benbo/WSGAN-paper. 

The StyleWSGAN model in this repository is based on StyleGAN2-ADA and its offical pytorch implementation provided by NVIDIA, which is available at https://github.com/NVlabs/stylegan2-ada-pytorch. StyleGAN2-ADA was proposed in:

**Training Generative Adversarial Networks with Limited Data**<br>
Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, Timo Aila<br>
https://arxiv.org/abs/2006.06676<br>


## Requirements
This repository retains the requirements of the original StyleGAN2-ADA implementation. Please refer to the original repository for details: https://github.com/NVlabs/stylegan2-ada-pytorch

Compared to StyleGAN2-ADA, this repository only requires the addition of torchmetrics (which could be removed quite easily if you desire to do so).

The `installation_notes.md` file contains an example to set up an environment which worked on our system.

## Preparing datasets

Datasets are stored as uncompressed ZIP archives containing: uncompressed PNG files, labeling function outputs as txt files (for each image), and a metadata file `dataset.json` for labels. LFs are assumed to output labels in {0,1,...C} where C is the number of classes and 0 denotes that an LF abstains from casting a vote. The code currently assumes that ground truth labels are available so that performance can be monitored. 

The `wsgan_dataset_tool.py` script uses torchvision datasets. 

**CIFAR-10**: 

Create a CIFAR10 ZIP archive with LFs used in our WSGAN paper. 
CIFAR10 will be downloaded if it is not found where `--data_path` points to. 


```.bash
# CIFAR10-B subset created for the WSGAN paper
python wsgan_dataset_tool.py --dataset CIFAR10-B --dest_root ~/datasets/ --data_path ~/downloads/cifar/
# CIFAR10-lownoise subset created for the WSGAN paper ablations
python wsgan_dataset_tool.py --dataset CIFAR10-lownoise --dest_root ~/datasets/ --data_path ~/downloads//cifar/
```

**LSUN**: 
First, download all 10 scene categories from the [LSUN project page](https://www.yf.io/p/lsun/).
Then, ensure that you have the `lmdb` Python package installed (e.g. `pip install lmdb`).
Run the following command to create the LSUN ZIP archive with LFs:

```.bash
python wsgan_dataset_tool.py --dataset LSUN --dest_root ~/datasets/ --data_path ~/downloads/lsun/raw
```

## Training a network from scratch

The following is an example command to train a new StyleWSGAN network from scratch on a weakly supervised lsun subset. Don't forget to update the paths: 

```.bash
python train.py --outdir=/mypath/training-runs --data=~/datasets/style/datasets/lsun_30lfs_256.zip --gpus=4 --cfg=lsun256 --cond=1
```

Or on a weakly supervised CIFAR10 subset
```.bash
python train.py --outdir=/mypath/training-runs --data=~/datasets/style/datasets/CIFAR10_25lfs.zip --gpus=2 --cfg=cifar --cond=1
```

If you are using a different dataset, you can largely follow StyleGAN2-ADA recommendations for new parameters. However, note that the `embed_features` argument is extremely important for the stability of StyleWSGAN's training dynamics, and is set lower than in StyleGAN2-ADA. Please also note that we disable style mixing and path length regularization in the experiments reported in our WSGAN paper. 

## License

Copyright &copy; 2021, NVIDIA Corporation. All rights reserved.

StyleWSGAN is derivative work. To the best of our ability, it is made available in accordance with, and under the original [Nvidia Source Code License](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html).
