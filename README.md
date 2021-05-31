## Image Colorisation with Gans

* Implementation of paper Image Colorisation using Generative Adverserial Networks in pytorch
* Model runs - [wandb experiment tracker](https://wandb.ai/veb-101/image-colorization?workspace=user-veb-101)
* Some basic details about training process. Apart from the default parameters described in the paper the changes are as follows:
  * Used Spectral normalization and Two-time Update rule.
  * Tried Self-Attention but failed due to memory issues
  * Two stage training in total 100 and 100 epochs each.
  * L Channed from LAB colorspace as input and 2 ab channels as output.
  * Training data consists of 10000 image subset from COCO 2014 training set..


* More in-depth details will be updated ASAP. The project has been completed.