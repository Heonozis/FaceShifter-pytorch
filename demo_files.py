import sys
import argparse

sys.path.append('./face_modules/')
from demo.lib import swap_faces
import cv2
import time


def write_image(path, img):
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    cv2.imwrite(path, img)

def process_image_files():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_image")
    parser.add_argument("target_image")
    parser.add_argument("output_image")
    args=parser.parse_args()

    print(args.source_image,"+",args.target_image,"->",args.output_image)

    Xs_raw = cv2.imread(args.source_image)
    Xt_raw = cv2.imread(args.target_image)
    s2t = swap_faces(Xs_raw, Xt_raw)
    write_image(args.output_image, s2t)

    #try:

        #s2t = swap_faces(Xs_raw, Xt_raw)
        #write_image(args.output_image)

    #except Exception as e:
    #    print 'Exception: ' + str(e)
    return

if __name__ == '__main__':
    process_image_files()
