import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.utils import save_image
import os, glob, cv2, time, copy, math
import numpy as np
from model import AutoEncoder_Seg, weights_init

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

class SegDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_list = glob.glob(os.path.join(self.image_dir, '*.png'))
        assert len(self.image_dir) > 0

    def __getitem__(self, index):
        img = cv2.cvtColor(cv2.imread(self.image_list[index]), cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img)

        mask = cv2.imread(self.image_list[index].replace('patch','mask'), cv2.IMREAD_GRAYSCALE)
        mask = transforms.ToTensor()(mask)
        return img, mask, self.image_list[index]

    def __len__(self):
        return len(self.image_list)


# Path
model_path = './Segmentation/checkpoint/AutoEncoder_Segmentation.pth'   # path of checkpoint to resume training from
mask_output_dir = './Segmentation/patch_output'
testset_dir = '/projects/patho1/Kechun/NestDetection/dataset/ROI/split/test'

# Dataset and Dataloader
data_dir = '/projects/patho1/Kechun/NestDetection/dataset/baseline/patch/val'
dataset = SegDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16)

def test_model(model):
    # Iterate over data.
    for idx, data in enumerate(dataloader):
        inputs, _, paths = data
        # print(inputs.size())
        # wrap them in Variable
        if torch.cuda.is_available():
            try:
                inputs = Variable(inputs.float().cuda())
            except:
                print('Exception in wrapping data in Variable! idx:{}'.format(idx))
        else:
            inputs = Variable(inputs)

        outputs = model(inputs)
        # save all masks
        outputs = outputs.to("cpu").data
        outputs = (outputs > 0.5).float()
        for i, path in enumerate(paths):
            save_image(outputs[i], os.path.join(mask_output_dir, os.path.basename(path)))
    return


def stitch_mask():
    test_img_list = glob.glob(os.path.join(testset_dir, '*.tif'))
    assert len(test_img_list) > 0
    test_patch_list = glob.glob(os.path.join(mask_output_dir, '*.png'))
    for img_path in test_img_list:
        bn = os.path.basename(img_path).split('.')[0]
        # patch path for this image
        patch_4img = [patch_path for patch_path in test_patch_list if bn in patch_path]
        # Get image shape
        original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        W, H = original_img.shape[0], original_img.shape[1]
        del original_img
        # Stitch it!
        full_mask = np.zeros(W, H)




# Model
model = AutoEncoder_Seg()
model = nn.DataParallel(model)
model.cuda()

assert model_path is not None
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model'])
model.eval()

test_model(model)

