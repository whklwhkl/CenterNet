import _init_paths

import cv2
import numpy as np

from flask import Flask, request, jsonify
from opts import opts
from detectors.pdet_no_disp import PersonDetector


opt = opts().init(['pdet'])
opt.task = 'pdet'
opt.load_model = 'exp/pdet/wider2019pd_raw_608_1216/dla34_ap567.pth'
detector = PersonDetector(opt)

app = Flask('detector')

@app.route('/det', methods=['POST'])
def predict():
    f = request.files['img']
    img = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_COLOR)
    ret = detector.run(img)
    boxes = []
    for box in ret['results'][1]:
        if box[-1] > .3:    # conf thresh
            boxes += [box.tolist()]
    return jsonify(boxes)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=6666)