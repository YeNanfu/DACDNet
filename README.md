# Accelerating Activity Inference on Edge Devices through Spatial Redundancy in Coarse-Grained Dynamic Networks

## If you think here any code component is useful, please cite this research.
@ARTICLE{10677444,
  author={Ye, Nanfu and Zhang, Lei and Xiong, Di and Wu, Hao and Song, Aiguo},
  journal={IEEE Internet of Things Journal}, 
  title={Accelerating Activity Inference on Edge Devices Through Spatial Redundancy in Coarse-Grained Dynamic Networks}, 
  year={2024},
  volume={11},
  number={24},
  pages={41273-41285},
  keywords={Convolutional neural networks;Delays;Human activity recognition;Hardware;Feature extraction;Termination of employment;Redundancy;Activity recognition;coarse-grained;delay prediction;dynamic convolution;mask;sensors},
  doi={10.1109/JIOT.2024.3458441}}

## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets
```
WISDM, UniMiB, PAMAOP2
```
### Environments

Environment details used for the main experiments. Every main experiment is conducted on a single RTX 3090 GPU.

```
Environment:
	Python: 3.8.18
	PyTorch: 1.12.1 
	Torchvision: 0.13.1
```



### File Description
models.resnet is the baseline network；

models.dacd_resnet is the proposed DACDNet；

models.utils_dacd is the necessary components of the DACDNet；

all python files in utils is the components for training such as loss function, optimizer and etc.
