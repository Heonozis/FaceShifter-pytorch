### FaceShifter
Implementation of https://arxiv.org/abs/1912.13457

### Reference:

https://github.com/taotaonice/FaceShifter


### Requirements

##### CUDA
[Cuda 10.1](https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130)

##### Packages
`pip install -r requirements.txt`

##### APEX
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Data
```
# get model weights
wget https://www.dropbox.com/s/kzo52d9neybjxsb/model_ir_se50.pth?dl=0 -O face_modules/model_ir_se50.pth
```

### Train
`python train_aei.py`

