import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import os, glob, cv2, time, copy, math
import numpy as np
from model import AutoEncoder, weights_init, weights_init_seg

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


model_path = './Segmentation/checkpoint/AutoEncoder_Segmentation.pth'   # path of checkpoint to resume training from
mask_output_dir = './Segmentation/patch_output'

# Dataset and Dataloader
data_dir = '/projects/patho1/Kechun/NestDetection/dataset/baseline/patch/val'
dataset = SegDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16)


def test_model(model):
    # Iterate over data.
    for idx, data in enumerate(dataloader):
        inputs, labels, paths = data
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
        outputs = outputs.to("cpu").data.numpy().astype(np.uint8)
        for i, path in enumerate(paths):
            cv2.imwrite(os.path.join(mask_output_dir, os.path.basename(path)), outputs[i, :, :, :].squeeze())
    return


def stitch_mask():
    # TODO
    pass


# Model
model = AutoEncoder()
model = nn.DataParallel(model)
model.cuda()

assert model_path is not None
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model'])
model.eval()

test_model(model)

