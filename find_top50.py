import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import json
import os
import sys
from torch.utils import data

torch.cuda.empty_cache()

is_parallel = True
if is_parallel:
    LEARNING_RATE = float(sys.argv[1])
else:
    LEARNING_RATE = 0.005

# LEARNING_RATE = 0.01, 0.005, 0.0005
CIFAR100_DIR = '/mnt/home/cchou/ceph/Data/cifar100_train_processed'
MODELS_PATH = f'/mnt/home/cchou/ceph/Capstone/CIFAR100_models/LR_{LEARNING_RATE}/'

# Ensure the save_path exists
# os.makedirs(OUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
])

IMAGE_DIM = 32
BATCH_SIZE = 500


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out



transform_train = transforms.Compose([
    transforms.Resize(IMAGE_DIM),
    transforms.ToTensor()
])

epoch = 59 #chose this as this model got highest train accuracy


trainset = datasets.ImageFolder(CIFAR100_DIR, transform_train)

dataloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
print("Length of dataloader", len(dataloader), flush=True)

top_50_map = {}
# model = ResNet9(3, 100)
with torch.no_grad():
    with open(MODELS_PATH + f'model_states_e{epoch}.pkl', 'rb') as file:
        model = ResNet9(3, 100)
        model_loaded = torch.load(file, map_location=torch.device('cpu'))
        model.load_state_dict(model_loaded)

for images, target in dataloader:
    # print(target)
    logits = torch.nn.functional.softmax(model(images), dim=-1)
    idx = target[0].item()
    print("index:", idx)
    values, indices = torch.sort(logits[:, idx], dim=0, descending=True)
    top_50_map[idx] = indices[:50].tolist()

with open(f"top_50_lr{LEARNING_RATE}.json", "w") as outfile:
    json.dump(top_50_map, outfile)
