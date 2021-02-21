import cv2
import numpy as np


class Mask:
    def __init__(self, mask_path, patch_size):
        self.map_code = {int('005500', 16): 1, int('ffffff', 16): 1, int('00ff00', 16): 1, int('aa00ff', 16): 1} 
        self.get_index(mask_path) # 2-D index mask
        self.patch_size = patch_size

    def get_index(self, path):
        mask = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        self.mask_X = mask.shape[0]
        self.mask_Y = mask.shape[1]

        mask_index = mask[:, :, 0] * 65536 + mask[:, :, 1] * 256 + mask[:, :, 2]
        # empty_mask: 1 indicates empty pixel
        self.empty_mask = (mask_index == int('808080', 16)).astype(int)
        mask_index = np.vectorize(self.map_code.get)(mask_index)
        # 0 as bg, positive number indicates different classes
        self.mask_index = np.nan_to_num(np.array(mask_index, dtype=np.float))

    def get_mask_patch(self, x, y):
        xmin = max(0, x)
        ymin = max(0, y)
        xmax = min(self.mask_X, x + self.patch_size)
        ymax = min(self.mask_Y, y + self.patch_size)
        if xmax - xmin == self.patch_size and ymax - ymin == self.patch_size:
            return self.mask_index[x:xmax, y:ymax]
        else:
            patch = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
            patch[xmin - x:xmax - x, ymin - y:ymax - y] = self.mask_index[xmin:xmax, ymin:ymax]
            return patch

    def check_empty(self, x, y):
        # return True if it's an empty patch
        xmin = max(0, x)
        ymin = max(0, y)
        xmax = min(self.mask_X, x + self.patch_size)
        ymax = min(self.mask_Y, y + self.patch_size)
        if xmax - xmin == self.patch_size and ymax - ymin == self.patch_size:
            empty_patch = self.empty_mask[x:xmax, y:ymax]
        else:
            empty_patch = np.ones((self.patch_size, self.patch_size))
            empty_patch[xmin - x:xmax - x, ymin - y:ymax - y] = self.empty_mask[xmin:xmax, ymin:ymax]
        empty_pixles = np.count_nonzero(empty_patch)
        if empty_pixles > 0.95 * (self.patch_size ** 2):
            return True
        return False

    def check_nest(self, x, y):
        # return True if it has nest in the patch
        xmin = max(0, x)
        ymin = max(0, y)
        xmax = min(self.mask_X, x + self.patch_size)
        ymax = min(self.mask_Y, y + self.patch_size)
        if xmax - xmin == self.patch_size and ymax - ymin == self.patch_size:
            mask_patch = self.mask_index[x:xmax, y:ymax]
        else:
            mask_patch = np.zeros((self.patch_size, self.patch_size))
            mask_patch[xmin - x:xmax - x, ymin - y:ymax - y] = self.mask_index[xmin:xmax, ymin:ymax]
        return np.count_nonzero(mask_patch) > 0
