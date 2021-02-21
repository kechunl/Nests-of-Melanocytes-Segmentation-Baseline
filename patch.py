import os, json, random, glob, math, cv2, argparse
from mask import *


def split_patch(img_path, mask_path, patch_img_dir, patch_mask_dir, patch_size=128, overlap=0.1):
	# For test set, overlap=0, this's for stitching patch.
	overlap_px = math.ceil(patch_size * overlap)

	img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
	mask = Mask(mask_path, patch_size=patch_size)

	bn = os.path.basename(img_path).split('.')[0]

	# get dimension of image
	img_X, img_Y = img.shape[0], img.shape[1]

	# counter of positive and negative samples
	positive = 0
	negative = 0
	overlap_positive = 0

	def extract_patch(x, y):
		xmin = max(0, x)
		ymin = max(0, y)
		xmax = min(img_X, x + patch_size)
		ymax = min(img_Y, y + patch_size)
		if xmax - xmin == patch_size and ymax - ymin == patch_size:
			return img[x:xmax, y:ymax, :]
		else:
			patch = 255 * np.ones((patch_size, patch_size, 3), dtype=np.uint8)
			patch[xmin-x:xmax - x, ymin-y:ymax - y, :] = img[xmin:xmax, ymin:ymax, :]
			return patch

	for x in range(0, img_X, patch_size):
		for y in range(0, img_Y, patch_size):
			patch_img = extract_patch(x, y)
			patch_mask = mask.get_mask_patch(x, y)
			# save patch and its mask
			patch_name = bn + '_{}_{}.png'.format(x, y)
			cv2.imwrite(os.path.join(patch_img_dir, patch_name), cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR))
			cv2.imwrite(os.path.join(patch_mask_dir, patch_name), patch_mask)
			# when there's nest in this patch, do the overlapped patchify
			if mask.check_nest(x, y):
				positive += 1
				if overlap != 0:	# do the overlapped patchify
					for overlap_x in range(x-int(patch_size/2), x+int(patch_size/2), overlap_px):
						if overlap_x > img_X:
							break
						for overlap_y in range(y-int(patch_size/2), y+int(patch_size/2), overlap_px):
							if overlap_y > img_Y:
								break
							if mask.check_nest(overlap_x, overlap_y):
								overlap_positive += 1
								patch_img = extract_patch(overlap_x, overlap_y)
								patch_mask = mask.get_mask_patch(overlap_x, overlap_y)
								# save patch and its mask
								patch_name = bn + '_{}_{}.png'.format(overlap_x, overlap_y)
								cv2.imwrite(os.path.join(patch_img_dir, patch_name), cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR))
								cv2.imwrite(os.path.join(patch_mask_dir, patch_name), patch_mask)
			else:
				negative += 1
	return [positive, overlap_positive, negative]


def main(args):
	img_dir = args.img_path
	mask_dir = args.mask_path
	dataset = args.set
	patch_size = args.patch_size
	nest_overlap = args.overlap
	patch_img_dir = args.patch_img_path
	patch_mask_dir = args.patch_mask_path

	os.makedirs(patch_img_dir, exist_ok=True)
	os.makedirs(patch_mask_dir, exist_ok=True)

	for d in dataset:
		os.makedirs(os.path.join(patch_img_dir, d), exist_ok=True)
		os.makedirs(os.path.join(patch_mask_dir, d), exist_ok=True)
		original_img_list = glob.glob(os.path.join(img_dir, d, '*.tif'))
		assert len(original_img_list) > 0
		print('Get {} images in dataset {}'.format(len(original_img_list), d))
		set_cnt = []	# count positive and negative samples in dataset
		for idx, img_path in enumerate(original_img_list):
			bn = os.path.basename(img_path).split('.')[0]
			# Find mask path
			mask_path = glob.glob(os.path.join(mask_dir, '**', bn+'.png'), recursive=True)
			if len(mask_path) == 0:
				print('*** Can not find the mask of {} , will SKIP this image ***'.format(bn+'.tif'))
				continue
			cnt = split_patch(img_path, mask_path[0], os.path.join(patch_img_dir, d), os.path.join(patch_mask_dir, d), patch_size=patch_size, overlap=nest_overlap)
			set_cnt.append(cnt)
			print('{}: images processed {}/{}. {}'.format(d, idx, len(original_img_list), bn))
		set_cnt = np.array(set_cnt)
		print('Total patches: {}\t Positive (with nest): {}\tNegative (w/o nest): {}\nP/N Ratio before overlap: {}\tRatio after overlap: {}\n'.format(np.sum(set_cnt), np.sum(set_cnt[:, 0:2]), np.sum(set_cnt[:, 2]), np.sum(set_cnt[:,0])/np.sum(set_cnt[:,2]), np.sum(set_cnt[:, 0:2])/np.sum(set_cnt[:, 2])))



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="patching WSI")

	parser.add_argument("--img_path", default='/projects/patho1/Kechun/NestDetection/dataset/ROI/split')
	parser.add_argument("--mask_path", default='/projects/patho1/Kechun/NestDetection/dataset/ROI/masks_plus_bg')
	# parser.add_argument("--set", default=('train', 'val',))
	parser.add_argument("--set", default=('test',))
	parser.add_argument("--patch_size", default=128)
	# parser.add_argument("--overlap", default=0.25)
	parser.add_argument("--overlap", default=0)
	parser.add_argument("--patch_img_path", default='/projects/patho1/Kechun/NestDetection/dataset/baseline/patch')
	parser.add_argument("--patch_mask_path", default='/projects/patho1/Kechun/NestDetection/dataset/baseline/mask')

	args = parser.parse_args()

	main(args)