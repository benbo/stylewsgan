# Setting up an environment for StyleWSGAN

```bash
conda create -n stylewsganada python=3.7
conda activate stylewsganada
# https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch
conda install pip
conda install psutil scipy
conda install -c conda-forge torchmetrics
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3 tensorboard
```
