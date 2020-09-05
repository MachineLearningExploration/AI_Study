This manuscript is for Windows.

## How to install PYTORCH

Before to install PYTORCH, you have to install these.

1. Python
2. Anaconda

Addictionally, if you want to use the GPU version, please install CUDA in advance.

3. Cuda(+CuDNN)


### Installation

At the ANACONDA COMMANDE, please type below script in the environment where you want to use for pytorch.

```
conda install pytorch torchvision cudotoolkit = 10.2 -c torch
```

This script is for me.
So based on your cuda version(Mine is 10.2), you have to revise the version.

### Check the installation

You can check the installation at python

```
import torch
```

After then, if you want to check the usage of GPU, you can check it based the below script.

```
torch.cuda.get_device_name(0)
torch.cuda.is_available()
```

If the result of `torch.cuda.is_available` is TRUE, it means GPU is working.
