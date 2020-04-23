import sys
sys.path.append('./face_modules/')
import torchvision.transforms as transforms
from face_modules.model import Backbone
from face_modules.mtcnn import *
import PIL.Image as Image
from network.aei import *
import numpy as np
import torch
import time
import cv2


class Screen_Capture:
    def __init__(self, H, W):
        self.H = H
        self.W = W
        self.cap = cv2.VideoCapture(0)

    def read_frame(self):
        ret, frame = self.cap.read()
        return np.array(frame)


screen_capture = Screen_Capture(1080, 960)

detector = MTCNN()
device = torch.device('cpu')
G = AEI_Net(c_id=512)
G.eval()
G.load_state_dict(torch.load('./saved_models/G_latest.pth', map_location=device))
G = G.cpu()

arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load('./saved_models/model_ir_se50.pth', map_location=device), strict=False)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


Xs_paths = ['source.jpg']
Xs_raws = [cv2.imread(Xs_path) for Xs_path in Xs_paths]
Xses = []
for Xs_raw in Xs_raws:
    try:
        Xs = detector.align(Image.fromarray(Xs_raw), crop_size=(64, 64))
        Xs = test_transform(Xs)
        Xs = Xs.unsqueeze(0).cpu()
        Xses.append(Xs)
    except:
        continue
Xses = torch.cat(Xses, dim=0)
with torch.no_grad():
    embeds, Xs_feats = arcface(F.interpolate(Xses[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))
    embeds = embeds.mean(dim=0, keepdim=True)


ind = 0

mask = np.zeros([64, 64], dtype=np.float)
for i in range(64):
    for j in range(64):
        dist = np.sqrt((i-32)**2 + (j-32)**2)/32
        dist = np.minimum(dist, 1)
        mask[i, j] = 1-dist
mask = cv2.dilate(mask, None, iterations=20)

cv2.namedWindow('image')#, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.moveWindow('image', 0, 0)
while True:
    try:
        Xt_raw = screen_capture.read_frame()
        Xt_raw = cv2.cvtColor(Xt_raw, cv2.COLOR_RGB2BGR)
    except:
        continue

    Xt, trans_inv = detector.align_fully(Image.fromarray(Xt_raw), crop_size=(64, 64),
                                         return_trans_inv=True, ori=[0,3,1])
    if Xt is None:
        cv2.imshow('image', Xt_raw)
        ind += 1
        cv2.waitKey(1)
        print('skip one frame')
        continue

    Xt = test_transform(Xt)

    Xt = Xt.unsqueeze(0).cpu()
    with torch.no_grad():
        st = time.time()
        Yt, _ = G(Xt, embeds)
        Yt = Yt.squeeze().detach().cpu().numpy()
        st = time.time() - st
        print(f'inference time: {st} sec')
        Yt = Yt.transpose([1, 2, 0])*0.5 + 0.5
        Yt = Yt
        Yt_trans_inv = cv2.warpAffine(Yt, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
        mask_ = cv2.warpAffine(mask,trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
        mask_ = np.expand_dims(mask_, 2)
        Yt_trans_inv = mask_*Yt_trans_inv + (1-mask_)*(Xt_raw.astype(np.float)/255.)

        merge = Yt_trans_inv

        cv2.imshow('image', merge)
        ind += 1
        cv2.waitKey(1)