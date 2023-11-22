import os
import time
import json
import torch
import torch.nn as nn
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


IMAGE_SIZE = 32
BATCH_SIZE = 512
NUM_EPOCHS = 150
LEARNING_RATE = 0.001

# CIFAR100_DIR = '/mnt/home/cchou/ceph/Data/'
CIFAR100_TRAIN_DIR = '/mnt/home/cchou/ceph/Data/cifar100_train_processed'
CIFAR100_TEST_DIR = '/mnt/home/cchou/ceph/Data/cifar100_test_processed'
OUTPUT_DIR = '/mnt/home/cchou/ceph/Capstone/'
CHECKPOINT_DIR = OUTPUT_DIR + '/CIFAR100_models_epoch150'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.ImageFolder(CIFAR100_TRAIN_DIR, transform_train)
testset = datasets.ImageFolder(CIFAR100_TEST_DIR, transform_test)
# trainset = datasets.CIFAR100(root=CIFAR100_DIR, train=True, download=True, transform=transform_train)
# testset = datasets.CIFAR100(root=CIFAR100_DIR, train=False, download=True, transform=transform_train)


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
    

model = ResNet9(3, 100)
model = model.to(device)


trainloader = data.DataLoader(
        trainset,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        batch_size=BATCH_SIZE)

testloader = data.DataLoader(
        testset,
        num_workers=2,
        drop_last=True,
        batch_size=BATCH_SIZE)

print('Dataloader created')


optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
print('Optimizer created')

criterion = nn.CrossEntropyLoss()

# Lists to store training and test errors
train_errors = []
test_errors = []


print('Starting training...')

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    start_time = time.time()
    for imgs, classes in trainloader:
        imgs, classes = imgs.to(device), classes.to(device)

        output = model(imgs)
        loss = criterion(output, classes)

        # update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
   

    # save checkpoints
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model_states_e{}.pkl'.format(epoch + 1))
    torch.save(model.state_dict(), checkpoint_path)
    
    # Calculate train error
    model.eval()
    correct_train = 0
    total_train = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

    train_acc = 100 * correct_train / total_train
    train_errors.append(100 - train_acc)
    
    # Calculate test error
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_acc = 100 * correct_test / total_test
    test_errors.append(100 - test_acc)
    
    end_time = time.time()
    
    results = {
        'train_errors': train_errors,
        'test_errors': test_errors
    }
    
    with open(OUTPUT_DIR+'train_test_errors.json', 'w') as json_file:
        json.dump[(results, json_file)
    print("Results saved in json")
    
        
    print(f"Epoch: {epoch + 1}, Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%", "Average train Loss:", total_loss/len(trainloader), "Time per epoch", end_time-start_time)

# Print the final errors
print("Train Errors:", train_errors)
print("Test Errors:", test_errors)
