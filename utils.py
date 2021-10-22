import os
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler, dataset
from torch.utils.data import DataLoader, TensorDataset

from models.vgg import  vgg16_bn
from models.resnet import resnet18, resnet34
from models.densenet import densenet121
from models.inception import inception_v3

from frequencyHelper import generateDataWithDifferentFrequencies_3Channel as freq_3t
from frequencyHelper import generateDataWithDifferentFrequencies_GrayScale as freq_t

from config import CIFAR_MEAN, CIFAR_STD, MNIST_MEAN, MNIST_STD

def get_loader(args, batch_size=None):

    ## Get Adversarial Data Loader

    if batch_size is None:
        batch_size = args.batch_size

    save_path = os.path.join('./adv_data', args.dataset + '/')
    save_path += args.model_name + '_' + args.attack + '.pt'

    adv_images, adv_labels = torch.load(save_path)
    adv_data = TensorDataset(adv_images.float()/255, adv_labels)
    adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=False)

    return adv_loader


def get_correct(outputs, labels):

    ## Correct Predicitons in a given batch

    _, pred = torch.max(outputs, 1)
    correct = (pred == labels).float().sum(0).item()
    return correct


def evaluate_attack(loader, model, args):

    ## Performance on a given attack

    print(f'Evaluating : {args.attack} on {args.model_name}')

    ## Go in eval mode
    model.eval()

    metrics = {'clean_acc':{'correct':0, 'total':0}, 'adv_acc':{'correct':0, 'total':0}}
    adv_loader = get_loader(args)
    pbar = tqdm.tqdm(zip(loader, adv_loader), unit="batches", leave=False, total=len(loader))

    for (data,labels),(adv_data, adv_labels) in pbar:
        
        data, adv_data, labels = data.to(args.device), adv_data.to(args.device), labels.to(args.device)
        
        clean_output = model(data)
        metrics['clean_acc']['correct'] += get_correct(clean_output, labels)
        metrics['clean_acc']['total']   += clean_output.size(0)

        adv_output = model(adv_data)
        metrics['adv_acc']['correct'] += get_correct(adv_output, labels)
        metrics['adv_acc']['total']   += adv_output.size(0)

    metrics['clean_acc']['acc'] = (metrics['clean_acc']['correct'] / metrics['clean_acc']['total']) * 100.
    metrics['adv_acc']['acc'] = (metrics['adv_acc']['correct'] / metrics['adv_acc']['total']) * 100.

    print(f'Clean Accuracy : {metrics["clean_acc"]["acc"]:.2f} \t|\t Correct : {metrics["clean_acc"]["correct"]} \t|\t Total : {metrics["clean_acc"]["total"]}')
    print(f'{args.attack} Accuracy : {metrics["adv_acc"]["acc"]:.2f} \t|\t Correct : {metrics["adv_acc"]["correct"]} \t|\t Total : {metrics["adv_acc"]["total"]}')

    return metrics


def get_freq(data, r, dataset='cifar10'):

    images = data.detach().cpu()

    if dataset == 'cifar10':
        images = images.permute(0,2,3,1)
        img_l, img_h = freq_3t(images, r=r)
        img_l, img_h = torch.from_numpy(np.transpose(img_l, (0,3,1,2))), torch.from_numpy(np.transpose(img_h, (0,3,1,2)))
        return img_l, img_h

    img_l, img_h = freq_t(images, r=r)
    img_l, img_h = torch.from_numpy(img_l).view(-1, 32, 32).unsqueeze(1), torch.from_numpy(img_h).view(-1, 32, 32).unsqueeze(1)
    return img_l, img_h


def load_data(root='./input/cifar10/', batch_size=32, valid_size=0.2, data=False):
    
    dataset = root.split('/')[-2]

    ## Normalization will go in model construction
    preprocess = transforms.Compose([transforms.Resize((32,32)),
                                        transforms.ToTensor()])
    
    ## Init the data
    if dataset == 'cifar10':
        print('Loading Cifar')
        train_data = torchvision.datasets.CIFAR10(root, train=True, transform=preprocess)
        test_data = torchvision.datasets.CIFAR10(root=root, train=False, transform=preprocess)
    elif dataset == 'mnist':
        print('Loading MNIST')
        train_data = torchvision.datasets.MNIST(root, train=True, transform=preprocess, download=True)
        test_data = torchvision.datasets.MNIST(root=root, train=False, transform=preprocess, download=True)
    elif dataset == 'fmnist':
        print('Loading FMNIST')
        train_data = torchvision.datasets.FashionMNIST(root, train=True, transform=preprocess, download=True)
        test_data = torchvision.datasets.FashionMNIST(root=root, train=False, transform=preprocess, download=True)
    else:
        print(f'{root} / {dataset} doesnt exist')

    ## Split Train and Valid Data
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size*num_train))
    train_idx,valid_idx = indices[split:],indices[:split]
    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
        
    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    if data:
        return test_data, test_loader
    
    return train_loader,valid_loader,test_loader


class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        if len(self.mean)>1:
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
        else:
            mean, std = self.mean, self.std
        return (input - mean) / std



def load_model(model_name, args):
    
    if args.dataset == 'cifar10':
        save_path = None
        MEAN, STD = CIFAR_MEAN, CIFAR_STD 
    else: 
        save_path = './'+args.dataset+'_state_dict/'+args.model_name+'.pt'
        MEAN, STD = MNIST_MEAN, MNIST_STD 

    if model_name == 'resnet18':
        base_model = resnet18(pretrained=True, save_path=save_path)
    elif model_name == 'resnet34':
        base_model = resnet34(pretrained=True, save_path=save_path)
    elif model_name == 'vgg16_bn':
        base_model = vgg16_bn(pretrained=True, save_path=save_path)
    elif model_name == 'densenet121':
        base_model = densenet121(pretrained=True)
    elif model_name == 'inception_v3':
        base_model = inception_v3(pretrained=True)
    else:
        print('MODEL NOT FOUND')

    
    norm_layer = Normalize(mean=MEAN, std=STD)

    model = nn.Sequential(
        norm_layer,
        base_model
    ).to(args.device)
        
    return model