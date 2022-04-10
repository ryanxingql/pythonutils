"""Crop all input images together.
python crop_two_images.py -img_paths <img1-path> <img2-path> ...

Cropping box is determined interactively.
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


def show_imgs(img_list, cap_list):
	for img, cap in zip(img_list, cap_list):
		cv2.imshow(cap, img)
	print("focus on image and press any key to continue")
	cv2.waitKey(0)
	cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument('-img_paths', type=str, nargs='+')
parser.add_argument('-save_dir', type=str, default='log')
parser.add_argument('-start_h', type=int, default=0)
parser.add_argument('-start_w', type=int, default=0)
parser.add_argument('-crop_h', type=int, default=100)
parser.add_argument('-crop_w', type=int, default=100)
args = parser.parse_args()

if not osp.exists(args.save_dir):
	os.mkdir(args.save_dir)

# read images
img_list = []
for img_path in args.img_paths:
	img_list.append(cv2.imread(img_path))
	if len(img_list) == 1:
		h, w, c = img_list[0].shape
		print(f"image shape: {h} {w} {c}")
	else:
		h2, w2, c2 = img_list[0].shape
		assert h2 == h and w2 == w and c2 == c

# add black box first, in case that we select the boundary area to add red box
img_black_bound_list = []
for img in img_list:
	img_black_bound = np.zeros(shape=(h+2, w+2, c), dtype=np.uint8)
	img_black_bound[1:1+h, 1:1+w, ...] = img
	img_black_bound_list.append(img_black_bound)

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
		crop_h = int(input("crop_h (default 100): ") or "100")
		crop_w = int(input("crop_w (default 100): ") or "100")

	img_red_box_list = []
	for img_black_bound in img_black_bound_list:
		img_red_box = add_red_box(img_black_bound, start_h+1, start_w+1, crop_h, crop_w)
		img_red_box_list.append(img_red_box)

	show_imgs(img_red_box_list, map(str, list(range(1, len(img_red_box_list)+1))))
	if_ok = input("OK? press y if OK: ")

# crop
for idx, img_black_bound in enumerate(img_black_bound_list):
	img_patch = img_black_bound[start_h:start_h+crop_h, start_w:start_w+crop_w, :]
	img_ori_name = args.img_paths[idx].split('/')[-1].split('.')[0]
	img_save_path = osp.join(args.save_dir, img_ori_name + f'_{start_h}_{start_w}_{crop_h}_{crop_w}_img{idx}.png')
	cv2.imwrite(img_save_path, img_patch)
	print(f'img {idx} saved at {img_save_path}')
