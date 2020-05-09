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
1. Get embedding model [weights](https://www.dropbox.com/s/kzo52d9neybjxsb/model_ir_se50.pth?dl=0)

2. Get CelebA [dataset](https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
) and extract to `../img_align_celeba`

3. Preprocess data
```
python face_modules/preprocess.py
```

### Train
`python train_aei.py`

Monitor `visdom` by visiting
```
https://HOST:8097
```

### Demo
1. Download [pretrained](https://drive.google.com/open?id=1sEalAgQKJKFHx-amY1eeIt-F97VLIQ4j) (45 epochs) `G_latest.pth` and `D_latest.pth` into `./saved_models` folder
2. Run `python demo_cam.py`
3. (optional) Change `./source.jpg` to change demo source image

### Todo
- Add HEAR-Net training and demo
