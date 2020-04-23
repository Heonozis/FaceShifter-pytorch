from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from torchvision import transforms as trans
import PIL.Image as Image
from mtcnn import MTCNN
import torch
import cv2
import os


img_root_dir = '../img_align_celeba'
save_path = '../celeba_64'

device = torch.device('cuda:0')
mtcnn = MTCNN()

model = Backbone(50, 0.6, 'ir_se').to(device)
model.eval()
model.load_state_dict(torch.load('./saved_models/model_ir_se50.pth'))

# threshold = 1.54
test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# decoder = libnvjpeg.py_NVJpegDecoder()

ind = 0
embed_map = {}

for root, dirs, files in os.walk(img_root_dir):
    for name in files:
        if name.endswith('jpg') or name.endswith('png'):
            try:
                p = os.path.join(root, name)
                img = cv2.imread(p)[:, :, ::-1]
                faces = mtcnn.align_multi(Image.fromarray(img), min_face_size=64, crop_size=(128, 128))
                if len(faces) == 0:
                    continue
                for face in faces:
                    scaled_img = face.resize((64, 64), Image.ANTIALIAS)
                    new_path = '%08d.jpg'%ind
                    ind += 1
                    print(new_path)
                    scaled_img.save(os.path.join(save_path, new_path))
            except Exception as e:
                continue
