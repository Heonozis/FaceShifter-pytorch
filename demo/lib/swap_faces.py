import torchvision.transforms as transforms
from face_modules.model import Backbone
from face_modules.mtcnn import *
import PIL.Image as Image
from network.aei import *
import numpy as np
import torch
import time
import cv2


def swap_faces(Xs_raw, Xt_raw):
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

    Xs_img = Image.fromarray(Xs_raw)
    Xs = detector.align(Xs_img, crop_size=(64, 64))

    Xs = test_transform(Xs)
    Xs = Xs.unsqueeze(0).cpu()
    with torch.no_grad():
        embeds, Xs_feats = arcface(F.interpolate(Xs, (112, 112), mode='bilinear', align_corners=True))
        embeds = embeds.mean(dim=0, keepdim=True)

    mask = np.zeros([64, 64], dtype=np.float)
    for i in range(64):
        for j in range(64):
            dist = np.sqrt((i-32)**2 + (j-32)**2)/32
            dist = np.minimum(dist, 1)
            mask[i, j] = 1-dist
    mask = cv2.dilate(mask, None, iterations=20)

    Xt_img = Image.fromarray(Xt_raw)

    Xt, trans_inv = detector.align(Xt_img, crop_size=(64, 64), return_trans_inv=True)

    Xt = test_transform(Xt)

    Xt = Xt.unsqueeze(0).cpu()
    with torch.no_grad():
        st = time.time()
        Yt, _ = G(Xt, embeds)
        Yt = Yt.squeeze().detach().cpu().numpy()
        st = time.time() - st
        print(f'inference time: {st} sec')
        Yt = Yt.transpose([1, 2, 0])*0.5 + 0.5
        Yt_trans_inv = cv2.warpAffine(Yt, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderMode=cv2.BORDER_TRANSPARENT)
        mask_ = cv2.warpAffine(mask, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderMode=cv2.BORDER_TRANSPARENT)
        mask_ = np.expand_dims(mask_, 2)
        Yt_trans_inv = mask_*Yt_trans_inv + (1-mask_)*(Xt_raw.astype(np.float)/255.)

        merge = Yt_trans_inv

        return merge
