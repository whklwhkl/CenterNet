import requests
import cv2
import io
from tqdm import tqdm
import json

HOST = '192.168.20.122'  # 192.168.20.122


DET_URL = 'http://%s:6666/det' % HOST


def det(img_file):
    response = requests.post(DET_URL, files={'img': img_file})
    return response.json()

def _nd2file(img_nd):
    return io.BytesIO(cv2.imencode('.jpg', img_nd)[1])

if __name__ == '__main__':
    from glob import glob

    import os
    # os.mkdir('chushimen')

    for im_path in tqdm(glob('/home/wanghao/Pictures/fixdata/*/*.jpg')):
        ret = []
        try:
            im = cv2.imread(im_path)
            boxes = det(_nd2file(im))
            for box in boxes:
                if box[-1] > .3:
                    l,t,r,b = map(int, box[:4])
                    cv2.rectangle(im, (l, t), (r,b), (0, 255, 0), 2)
                ret += [{'ltrb': [l,t,r,b]}]
            # cv2.imwrite('chushimen/' + os.path.split(im_path)[1], im)
            json.dump({'ret': ret}, open('chushimen/' + os.path.split(im_path)[1].replace('.jpg', '.json'), 'w'), indent=2)
        except:
            print(os.path.split(im_path)[1])
