import logging

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info(f'is torch available = {torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'device = {device}')

# device_id = torch.cuda.current_device()
# print(device_id)
# print(torch.cuda.get_device_name(device_id))
# print(torch.cuda.memory_allocated(device_id))
#
# X_train = torch.FloatTensor([0., 1., 2.])
# print(X_train)
# print(X_train.is_cuda)
# X_train = X_train.to(device)
# print(X_train)
# print(X_train.is_cuda)

"""
Cuda Installation
=======================
https://askubuntu.com/questions/1230645/when-is-cuda-gonna-be-released-for-ubuntu-20-04
https://askubuntu.com/questions/927199/nvidia-smi-has-failed-because-it-couldnt-communicate-with-the-nvidia-driver-ma

Cuda pytorch running
https://pytorch.org/docs/stable/cuda.html
https://deeplizard.com/learn/video/Bs1mdHZiAS8
https://pytorch.org/docs/stable/notes/cuda.html
https://towardsdatascience.com/pytorch-switching-to-the-gpu-a7c0b21e8a99

Steps
1) make sure nvidia-smi is installed
2) disable secure boot
"""
