from cv2 import cv2

def bgr2rgb(img):
    code = getattr(cv2, 'COLOR_BGR2RGB')
    img = cv2.cvtColor(img, code)
    return img

def rgb2bgr(img):
    code = getattr(cv2, 'COLOR_RGB2BGR')
    img = cv2.cvtColor(img, code)
    return img
