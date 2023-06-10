import PIL
import torch
import torchvision
import cv2
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pandas as pd
from torchvision.io import read_image
import transforms
from PIL import Image


dataset_path = 'fer2013/fer2013/fer2013.csv'
image_size = (48, 48)
max_sample = 500
folder_path = './data/FERPlus/'
training_path = 'FER2013Train'
test_path = 'FER2013Test'
file_extension = 'png'
csv_name = 'label.csv'
result = []


class MNIST(object):
    def __init__(self, batch_size, use_gpu, num_workers):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        pin_memory = True if use_gpu else False

        trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 10


def format_img(img):
    img = cv2.resize(img.astype('uint8'), (224, 224), interpolation=cv2.INTER_CUBIC)
    img.astype('float32')
    return img


def reshape(arr):
    if not isinstance(arr, type(np.ones((2, 2)))):
        raise Exception("请传入ndarray对象")
    res = []
    for i in arr:
        i = format_img(i)
        res.append(i)
    res = np.array(res)
    res = np.expand_dims(res, -1)
    res = np.repeat(res, 3, axis=-1)
    return res


class FERPlusDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image = image.numpy()
        image = reshape(image)
        image = image.squeeze(0)
        image = Image.fromarray(np.uint8(image), mode='RGB')
        label = torch.from_numpy(self.img_labels.iloc[idx, 2:10].to_numpy(dtype='float'))
        return self.transform(image), label

class FERPlus(object):
    def __init__(self, batch_size, use_gpu, num_workers):

        pin_memory = True if use_gpu else False

        trainset = FERPlusDataset(folder_path + training_path + '/label.csv', folder_path + training_path,
                                  transform=transforms.ToTensor())

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = FERPlusDataset(folder_path + test_path + '/label.csv', folder_path + test_path,
                                 transform=transforms.ToTensor())

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 8


__factory = {
    'mnist': MNIST,
    'ferplus': FERPlus
}


def create(name, batch_size, use_gpu, num_workers):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](batch_size, use_gpu, num_workers)
