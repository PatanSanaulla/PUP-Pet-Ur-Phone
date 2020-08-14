# Python 2/3 compatibility
from __future__ import print_function
import requests
import numpy as np
import imutils
import cv2
import sys

# Global vars:
WIDTH = 700
STEP = 16
QUIVER = (255, 100, 0)
url = "http://10.105.161.8:8080/shot.jpg"


def draw_flow(img, flow, step=STEP):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, QUIVER)
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


if __name__ == '__main__':
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=WIDTH)
    prevgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cam = cv2.VideoCapture(fn)
    # ret, prev = cam.read()
    # prev = imutils.resize(prev, width=WIDTH)
    # prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        # ret, img = cam.read()
        # img = imutils.resize(img, width=WIDTH)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=WIDTH)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray

        cv2.imshow('flow', draw_flow(gray, flow))

        ch = cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows()