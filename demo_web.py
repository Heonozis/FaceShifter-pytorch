import sys
sys.path.append('./face_modules/')
from demo.lib import swap_faces
import cv2
import time

from flask import render_template, request, Flask
app = Flask(__name__, template_folder='demo/templates', static_url_path='', static_folder='demo/static')


def write_image(path, img):
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    cv2.imwrite(path, img)


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html', )


@app.route('/', methods=["GET", "POST"])
def process():
    
    id = str(int(time.time()))

    if request.method == 'GET':
        return render_template('index.html')

    source = request.files['source']
    source.save(f'./demo/static/source_{id}.jpg')

    target = request.files['target']
    target.save(f'./demo/static/target_{id}.jpg')

    Xs_raw = cv2.imread(f'./demo/static/source_{id}.jpg')
    Xt_raw = cv2.imread(f'./demo/static/target_{id}.jpg')

    try:

        s2t = swap_faces(Xs_raw, Xt_raw)
        write_image(f'./demo/static/result_s2t_{id}.jpg', s2t)

        Xs_raw = cv2.imread(f'./demo/static/target_{id}.jpg')
        Xt_raw = cv2.imread(f'./demo/static/source_{id}.jpg')

        t2s = swap_faces(Xs_raw, Xt_raw)
        write_image(f'./demo/static/result_t2s_{id}.jpg', t2s)
    except Exception as e:
        error = 'Exception: ' + str(e)
        return render_template('index.html', error=error)


    data = {
        "source": f"source_{id}.jpg",
        "target": f"target_{id}.jpg",
        "result_s2t": f"result_s2t_{id}.jpg",
        "result_t2s": f"result_t2s_{id}.jpg"
    }

    return render_template('index.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)
