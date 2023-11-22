import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
import numpy as np
import os
import sys
import json


torch.cuda.empty_cache()

CIFAR100_DIR = '/mnt/home/cchou/ceph/Data/cifar100_train_processed'
MODELS_PATH = '/mnt/home/cchou/ceph/Capstone/CIFAR100_models_epoch150'
OUTPUT_DIR = '/mnt/home/cchou/ceph/Capstone/CIFAR100_jsons/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_DIM = 32
BATCH_SIZE = 500
DIM = 2000


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


is_parallel = True
if is_parallel:
    epoch = int(sys.argv[1])
else:
    epoch = 15

layer_X_projected = {}


transform_train = transforms.Compose([
    transforms.Resize(IMAGE_DIM),
    transforms.ToTensor()
])

trainset = datasets.ImageFolder(CIFAR100_DIR, transform_train)

dataloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
print("Length of dataloader", len(dataloader), flush=True)


with torch.no_grad():
    with open(MODELS_PATH + f'model_states_e{epoch}.pkl', 'rb') as file:
        model = ResNet9(3, 100)
        model_loaded = torch.load(file, map_location=torch.device('cpu'))
        model.load_state_dict(model_loaded)

    print("Model Loaded", flush=True)
    layers_list = list(model.children())

    for layer in range(len(layers_list)):
        print("Layer:", layer, flush=True)
        clipped_model = nn.Sequential(*layers_list[:layer])
        X_projected = []
        M_tag = None

        for images, target in dataloader:
            output = clipped_model(images)  # clip the layers for the class images
            X_i = output.view(-1, images.shape[0])  #flatten the images,
            X_i = X_i.detach().numpy()
            dim_representation = X_i.shape[0]
            if M_tag == None:
                M = np.random.randn(DIM, dim_representation)
                M /= np.sqrt(np.sum(M * M, axis=1, keepdims=True))
                M_tag = "Done"
            X_i = (M @ X_i)
            X_projected.append(X_i.tolist())

        print(f"Projecting done: {layer}", flush=True)

        # X_projected should be a list of numpy arrays and each array will be (N * 500) projected on (3000 * 500)
        layer_X_projected[layer+1] = X_projected

    print("Saving", flush=True)

with open(f'{OUTPUT_DIR}/epoch_{epoch}.json', 'w') as json_file:
    json.dump(layer_X_projected, json_file)