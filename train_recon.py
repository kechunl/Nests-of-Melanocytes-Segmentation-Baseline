import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import os, glob, cv2, time, copy, math
from model import AutoEncoder, weights_init

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Tensorboard Writer
task = 'Reconstruction'
os.makedirs(os.path.join('./', task), exist_ok=True)
train_writer = SummaryWriter(os.path.join('.', task, 'train'))
val_writer = SummaryWriter(os.path.join('.', task, 'val'))

class ReconDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_list = glob.glob(os.path.join(self.image_dir, '*.png'))
        assert len(self.image_dir) > 0

    def __getitem__(self, index):
        img = cv2.cvtColor(cv2.imread(self.image_list[index]), cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img)
        return img

    def __len__(self):
        return len(self.image_list)


num_epochs = 100
batch_size = {'train': 256, 'val': 64}
BASE_LR = 1e-3
RESUME = None   # path of checkpoint to resume training from
print_freq = {'train': 400, 'val': 1000}

model_save_dir = os.path.join('.', task, 'checkpoint')
os.makedirs(model_save_dir, exist_ok=True)

# Dataset and Dataloader
data_dir = '/projects/patho1/Kechun/NestDetection/dataset/baseline/patch'
dsets = {x: ReconDataset(os.path.join(data_dir, x)) for x in ['train', 'val']}
dset_loaders = {x: DataLoader(dsets[x], batch_size=batch_size[x], shuffle=True, num_workers=16) for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}

def train_model(model, criterion, optimizer, lr_scheduler, start_epoch=0, num_epochs=100):
    since = time.time()

    best_model = model
    best_mse = 1e6

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()

            running_loss = 0.0
            epoch_loss = 0.0

            # Iterate over data.
            for idx, data in enumerate(dset_loaders[phase]):
                inputs = data
                # print(inputs.size())
                # wrap them in Variable
                if torch.cuda.is_available():
                    try:
                        inputs = Variable(inputs.float().cuda())
                    except:
                        print('Exception in wrapping data in Variable! idx:{}'.format(idx))
                else:
                    inputs = Variable(inputs)

                # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, inputs)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # print evaluation statistics
                try:
                    epoch_loss += outputs.shape[0] * loss.item()
                    running_loss += loss.item()
                except:
                    print('unexpected error, could not calculate loss or do a sum.')
                if (idx+1) % print_freq[phase] == 0:
                    print('[{}] Epoch: {} Iter {}/{} loss: {:.4f} lr: {}'.format(phase, epoch, idx, len(dset_loaders[phase]), running_loss/(idx+1), optimizer.state_dict()['param_groups'][0]['lr']))
            epoch_loss = epoch_loss / dset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val':
                val_writer.add_scalar('Loss', epoch_loss, epoch)
                if epoch_loss < best_mse:
                    best_mse = epoch_loss
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch
                    print('new best mse = ', best_mse)
                    # Save Model
                    save_files = {'model': best_model.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'lr_scheduler': lr_scheduler.state_dict(),
                                  'epoch': best_epoch}
                    torch.save(save_files, os.path.join(model_save_dir, "AutoEncoder_{}.pth".format(task)))
            elif phase == 'train':
                train_writer.add_scalar('Loss', epoch_loss, epoch)
                train_writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        lr_scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val MSE: {:4f}'.format(best_mse))
    return


# Model
model = AutoEncoder()
model.apply(weights_init)
model = nn.DataParallel(model)
model.cuda()

# criterion, optimizer, lr_scheduler
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

if RESUME != None:
    checkpoint = torch.load(RESUME)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    print('The training process from epoch {}...'.format(start_epoch))
else:
    start_epoch = 0

train_model(model, criterion, optimizer, lr_scheduler, start_epoch=start_epoch, num_epochs=num_epochs)

