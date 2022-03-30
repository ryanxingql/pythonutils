"""Crop two images together according to the axis.
python crop_two_images.py -img1 <img1-path> -img2 <img2-path>
"""
from cv2 import cv2
import numpy as np
import argparse
import os.path as osp
import os


def add_red_box(img, start_h, start_w, h, w):
	img = img.copy()

	# top bar
	img[start_h-1, start_w-1:start_w+1+w, 0] = 0
	img[start_h-1, start_w-1:start_w+1+w, 1] = 0
	img[start_h-1, start_w-1:start_w+1+w, 2] = 255

	# bottom bar
	img[start_h+1+h, start_w-1:start_w+1+w, 0] = 0
	img[start_h+1+h, start_w-1:start_w+1+w, 1] = 0
	img[start_h+1+h, start_w-1:start_w+1+w, 2] = 255

	# left bar
	img[start_h-1:start_h+1+h, start_w-1, 0] = 0
	img[start_h-1:start_h+1+h, start_w-1, 1] = 0
	img[start_h-1:start_h+1+h, start_w-1, 2] = 255

	# right bar
	img[start_h-1:start_h+1+h, start_w+1+w, 0] = 0
	img[start_h-1:start_h+1+h, start_w+1+w, 1] = 0
	img[start_h-1:start_h+1+h, start_w+1+w, 2] = 255
	return img


def show_two_imgs(img1, img2, cap1='img1', cap2='img2'):
	cv2.imshow(cap1, img1)
	cv2.imshow(cap2, img2)
	print("focus on image and press any key to continue")
	cv2.waitKey(0)
	cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument('-img1', type=str)
parser.add_argument('-img2', type=str)
parser.add_argument('-save_dir', type=str, default='log')
parser.add_argument('-start_h', type=int, default=0)
parser.add_argument('-start_w', type=int, default=0)
parser.add_argument('-crop_h', type=int, default=100)
parser.add_argument('-crop_w', type=int, default=100)
args = parser.parse_args()

if not osp.exists(args.save_dir):
	os.mkdir(args.save_dir)

# read images
img1 = cv2.imread(args.img1)
h, w, c = img1.shape
img2 = cv2.imread(args.img2)
h2, w2, c2 = img2.shape
assert h == h2 and w == w2 and c == c2
print(f"image shape: {h} {w} {c}")

# add black box first, in case that we select the boundary area to add red box
img1_black_bound = np.zeros(shape=(h+2, w+2, c), dtype=np.uint8)
img1_black_bound[1:1+h, 1:1+w, ...] = img1
img2_black_bound = np.zeros(shape=(h+2, w+2, c), dtype=np.uint8)
img2_black_bound[1:1+h, 1:1+w, ...] = img2
show_two_imgs(
	img1_black_bound,
	img2_black_bound,
	'img1 with black boundary',
	'img2 with black boundary',
)

if_first = True
while if_first or (if_ok != 'y'):
	if if_first:
		start_h = args.start_h
		start_w = args.start_w
		crop_h = args.crop_h
		crop_w = args.crop_w
		if_first = False

	else:
		start_h = int(input("start_h: "))
		start_w = int(input("start_w: "))
		crop_h = int(input("crop_h: "))
		crop_w = int(input("crop_w: "))

	img1_red_box = add_red_box(img1_black_bound, start_h+1, start_w+1, crop_h, crop_w)
	img2_red_box = add_red_box(img2_black_bound, start_h+1, start_w+1, crop_h, crop_w)
	show_two_imgs(
		img1_red_box,
		img2_red_box,
		'img1 with red box',
		'img2 with red box',
	)
	if_ok = input("OK? press y if OK: ")

img1_patch = img1[start_h:start_h+crop_h, start_w:start_w+crop_w, :]
img2_patch = img2[start_h:start_h+crop_h, start_w:start_w+crop_w, :]
img1_ori_name = args.img1.split('/')[-1].split('.')[0]
img2_ori_name = args.img2.split('/')[-1].split('.')[0]
img1_save_path = osp.join(args.save_dir, img1_ori_name + f'_{start_h}_{start_w}_{crop_h}_{crop_w}_img1.png')
img2_save_path = osp.join(args.save_dir, img2_ori_name + f'_{start_h}_{start_w}_{crop_h}_{crop_w}_img2.png')
cv2.imwrite(img1_save_path, img1_patch)
cv2.imwrite(img2_save_path, img2_patch)
print(f'img1 saved at {img1_save_path}')
print(f'img2 saved at {img2_save_path}')
