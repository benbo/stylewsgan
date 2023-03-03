# StyleWSGAN

**Coming soon, stay tuned**

This repository will contain additional code to reproduce ablation experiments presented in:

Generative Modeling Helps Weak Supervision (and Vice Versa)
Benedikt Boecking, Nicholas Roberts, Willie Neiswanger, Stefano Ermon, Frederic Sala, Artur Dubrawski
International Conference on Learning Representations (ICLR) (2023)
[arXiv] [OpenReview]

With the code in this repository, a weakly supervised GAN (WSGAN) can be trained on weakly supervised images, using a StyleGAN2 model as its base network. The WSGAN, which consists of a GAN and a label model, is trained on image datasets where some of the images have one or more labeling function (LF) votes associated that output weak labels, following the programmatic weak supervision/data programming paradigm. The code for the main experiments presented in the paper above, where a simple DCGAN is used as the base network, can be found at https://github.com/benbo/WSGAN-paper.
