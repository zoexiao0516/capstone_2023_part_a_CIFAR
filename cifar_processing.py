import os
import torch
from torchvision import datasets, transforms

# Set the directory where you want to save the CIFAR-100 dataset
# download_path = '/Users/shreemayi/Desktop/MSDS/CapStone/capstone_2023_part_a_CIFAR/data'
download_path = '/mnt/home/cchou/ceph/Data/'

# Create a directory to save the dataset
os.makedirs(download_path, exist_ok=True)

# Define a transformation to normalize the images
transform = transforms.Compose([
    transforms.ToTensor()
])

# Download the CIFAR-100 dataset
cifar100_dataset = datasets.CIFAR100(download_path, train=True, download=True, transform=transform)

# Create a directory to store grouped images
# save_path = '/Users/shreemayi/Desktop/MSDS/CapStone/capstone_2023_part_a_CIFAR/data/cifar100_processed/'
save_path = '/mnt/home/cchou/ceph/Data/cifar100_train_processed/'

# Create subdirectories for each class
for class_idx, class_name in enumerate(cifar100_dataset.classes):
    class_dir = os.path.join(save_path, class_name)
    os.makedirs(class_dir, exist_ok=True)

counters = [-1 for i in range(100)]
# Group and save the images into class folders
for i, (image, target) in enumerate(cifar100_dataset):
    class_name = cifar100_dataset.classes[target]
    counters[target] += 1
    class_dir = os.path.join(save_path, class_name)
    image_filename = f"{counters[target]}.png"
    # Convert the tensor image to a PIL image
    image_pil = transforms.ToPILImage()(image)
    image_pil.save(os.path.join(class_dir, image_filename))

print("CIFAR-100 dataset has been downloaded and images are grouped by class.")