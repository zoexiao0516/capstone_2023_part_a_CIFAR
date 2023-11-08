import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import sys
from torch.utils import data

torch.cuda.empty_cache()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)


CIFAR100_DIR = '/mnt/home/cchou/ceph/Data/cifar100_train_processed'
MODELS_PATH = '/mnt/home/cchou/ceph/Capstone/CIFAR100_models/'
OUT_DIR = '/mnt/home/cchou/ceph/Capstone/CIFAR100_Dataframes/'
# Ensure the save_path exists
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_DIM = 32
BATCH_SIZE = 500
# DIM_list = [1000, 1500, 2000]
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

rows = []


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(IMAGE_DIM),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
            # print(images.shape, target) 
            output = clipped_model(images)  # clip the layers for the class images
            X_i = output.view(-1, images.shape[0])  # flatten the images, X_i is of (flatted image representation dim, 500)
            X_i = X_i.detach().numpy()
            # print("X_i shape", X_i.shape, flush=True) X_i shape (3072, 500)
            dim_representation = X_i.shape[0]
            if M_tag == None:
                M = np.random.randn(DIM, dim_representation)
                M /= np.sqrt(np.sum(M * M, axis=1, keepdims=True))
                # M = M.to(device)
                M_tag = "Done"
            X_i = (M@X_i)
            X_projected.append(X_i)

        print(f"Projecting done: {layer}", flush=True)
            
        # X_projected should be a list of numpy arrays and each array will be (N * 500) projected on (3000 * 500)
            

        epoch_layer_data = {'epoch': epoch, 'layer': layer, 'X_projected': X_projected}
        rows.append(epoch_layer_data)


    print("Saving", flush=True)
    df = pd.DataFrame.from_records(rows)
    print(df.head(5), flush=True)
    df.to_pickle(OUT_DIR+'epoch'+str(epoch)+'.pkl')