### FaceShifter
Implementation of https://arxiv.org/abs/1912.13457

### Reference:

https://github.com/taotaonice/FaceShifte


### Install APEX
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Train
```
# get model weights
wget https://www.dropbox.com/s/kzo52d9neybjxsb/model_ir_se50.pth?dl=0 -O face_modules/model_ir_se50.pth
```

