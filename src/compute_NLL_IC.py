import argparse
import random
import os
import sys
import json
import time

from addict import Dict
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import cv2

import utils
import DCGAN_VAE_pixel as DVAE



class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return (img,index)


def KL_div(mu,logvar,reduction = 'none'):
    mu = mu.view(mu.size(0),mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    
    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1) 
        return KL


def store_NLL(x, recon, mu, logvar, z):
    with torch.no_grad():
        sigma = torch.exp(0.5*logvar)
        b = x.size(0)
        
        target = Variable(x.data.view(-1) * 255).long()
        recon = recon.contiguous()
        recon = recon.view(-1,256)
        cross_entropy = F.cross_entropy(recon, target, reduction='none')
        log_p_x_z = -torch.sum(cross_entropy.view(b ,-1), 1)
        
        log_p_z = -torch.sum(z**2/2+np.log(2*np.pi)/2,1)
        z_eps = (z - mu) / sigma
        z_eps = z_eps.view(b,-1)
        log_q_z_x = -torch.sum(z_eps**2/2 + np.log(2*np.pi)/2 + logvar/2, 1)
        
        weights = log_p_x_z+log_p_z-log_q_z_x
        
    return weights

def compute_NLL(weights):
    with torch.no_grad():
        NLL_loss = -(torch.log(torch.mean(torch.exp(weights - weights.max())))+weights.max())
        
    return NLL_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, help="directory of output from training")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--batchSize", type=int, default=1)
    parser.add_argument('--repeat', type=int, default=200, help='repeat for comute IWAE bounds')
    
    parser.add_argument('--num_iter', type=int, default=100, help='number of iters to optimize')
    parser.add_argument('--lr', type=float, default=2e-4, help='adam learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    
    parser.add_argument('--ic_type', default='png', help='type of complexity measure, choose between png and jp2')
    
    opt = parser.parse_args()
    
    return opt


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def main():
    if not torch.cuda.is_available():
        print("no gpu device available")
        sys.exit(1)
        
    test_opt = parse_args()
    with open(os.path.join(test_opt.train_dir, "opt.json")) as f:
        opt = Dict(json.load(f))
    # merge training and test params, overwrite training param with test param if overlapping
    for key, value in vars(test_opt).items():
        opt[key] = value
    
    device = "cuda:0"
    
    init_seeds(opt.seed, False)
    
    opt.savedir = "{}/test-{}".format(opt.train_dir, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(opt.savedir, scripts_to_save=None)
    with open(os.path.join(opt.savedir, "opt.json"), "w") as f:
        json.dump(opt, f, indent=4)
    
    transform = transforms.Compose([transforms.Resize((opt.imageSize)), transforms.CenterCrop((opt.imageSize)), transforms.ToTensor()])
    # setup dataset
    if opt.dataset == "fmnist":
        test_data = dset.FashionMNIST(root=opt.datadir, train=False, download=True, transform=transform)
    elif opt.dataset == "mnist":
        test_data = dset.MNIST(root=opt.datadir, train=False, download=True, transform=transform)
    elif opt.dataset == "cifar10":
        test_data = dset.CIFAR10(root=opt.datadir, download=True, train=False, transform=transform)
    elif opt.dataset == "svhn":
        test_data = dset.SVHN(root=opt.datadir, download=True, split="test", transform=transform)
    elif opt.dataset == "lsun":
        test_data = dset.LSUN(root=opt.datadir, classes="test", transform=transform)
    elif opt.dataset == "celeba":
        test_data = dset.CelebA(root=opt.datadir, split="test", download=True, transform=transform)
    elif opt.dataset == "cars":
        test_data = dset.StanfordCars(root=opt.datadir, split="test", download=True, transform=transform)
    elif opt.dataset == "country":
        test_data = dset.Country211(root=opt.datadir, split="test", download=True, transform=transform)
    elif opt.dataset == "kmnist":
        test_data = dset.KMNIST(root=opt.datadir, train=False, download=True, transform=transform)
    elif opt.dataset == "omniglot":
        test_data = dset.Omniglot(root=opt.datadir, background=False, download=True, transform=transform)
    
    test_queue = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    
    
    # setup models
    netG = DVAE.DCGAN_G(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu).to(device)
    netE = DVAE.Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu).to(device)
    ckpt = torch.load(os.path.join(opt.train_dir, "best.ckpt"))
    
    # ----------------------------------------------- #
    # netG.load_state_dict(ckpt["netG"])
    # netE.load_state_dict(ckpt["netE"])
    # need to set strict=False so that the channel increase does not cause error
    netG.load_state_dict(ckpt["netG"], strict=False)
    netE.load_state_dict(ckpt["netE"], strict=False)
    netG.eval()
    netE.eval()
    
    # setup loss function
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    
    def nll_helper(netE):
        weights_agg = []
        
        with torch.no_grad():
            x = xi.expand(opt.repeat, -1, -1, -1).contiguous()
            if x.shape[1] == 1:
                x = x.expand(-1, 3, -1, -1)
            
            if x.shape[1] == 3:
                # ----------------------------------------------- #
                # add in contour: gray(orig) - gray(Gaussian(orig))
                gray_orig = transforms.Grayscale(num_output_channels=1)(x)
                gaus_orig = transforms.GaussianBlur(kernel_size=9, sigma=3)(x)
                gray_blur = transforms.Grayscale(num_output_channels=1)(gaus_orig)
                contour = (gray_orig - gray_blur+0.5).clamp(0, 1)
                x = torch.cat((contour, x), dim=1).detach()
            
            for batch_number in range(5):
                x = x.to(device, non_blocking=True)                
                b = x.size(0)
                
                [z, mu, logvar] = netE(x)
                recon = netG(z)
                # print(z.shape, mu.shape, logvar.shape)
                mu = mu.view(mu.size(0), mu.size(1))
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0), z.size(1))
                weights = store_NLL(x, recon, mu, logvar, z)
                
                weights_agg.append(weights)
            weights_agg = torch.stack(weights_agg).view(-1) # (1000,)
            
            nll = compute_NLL(weights_agg)
        
        return nll
    
    NLL, NLL_IC = [], []
    
    for i, (xi, _) in enumerate(test_queue):
        x = xi.expand(opt.repeat, -1, -1, -1).contiguous()
        
        # ----------------------------------------------- #
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        # add in contour: gray(orig) - gray(Gaussian(orig))
        gray_orig = transforms.Grayscale(num_output_channels=1)(x)
        gaus_orig = transforms.GaussianBlur(kernel_size=9, sigma=3)(x)
        gray_blur = transforms.Grayscale(num_output_channels=1)(gaus_orig)
        contour = gray_orig - gray_blur
        contour = (contour+0.5).clamp(0, 1)
        x = torch.cat((contour, x), dim=1).detach()
        
        # compute the negative log-likelihood before optimizing q(z|x)
        NLL_loss = nll_helper(netE).detach().cpu().numpy()
        NLL = np.append(NLL, NLL_loss)

        img = x[0][:-1].permute(1,2,0)
        img = img.detach().cpu().numpy()
        img *= 255
        img = img.astype(np.uint8)
        if opt.ic_type == 'jp2':
            img_encoded=cv2.imencode('.jp2',img)
        elif opt.ic_type == 'png':
            img_encoded=cv2.imencode('.png',img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
        else:
            raise NotImplementedError("choose ic type between jp2 and png")
        IC = len(img_encoded[1])*8
        NLL_IC = np.append(NLL_IC, NLL_loss - IC)
        
        print(f"Image {i:d} [before] {NLL_loss:.3f} [after] {(NLL_loss - IC):.3f} [diff] {IC:.3f}")
        if i >= 1000: # test for 1000 samples
            break
    np.save(os.path.join(opt.savedir, f"NLL.npy"), NLL)
    np.save(os.path.join(opt.savedir, f"NLL_IC.npy"), NLL_IC)
    

if __name__ == "__main__":
    main()