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
    
    transform = transforms.Compose([transforms.Resize((opt.imageSize)), transforms.ToTensor()])
    # setup dataset
    if opt.dataset == "fmnist":
        test_data = dset.FashionMNIST(root=opt.datadir, train=False, download=True, transform=transform)
    elif opt.dataset == "mnist":
        test_data = dset.MNIST(root=opt.datadir, train=False, download=True, transform=transform)
    elif opt.dataset == "cifar10":
        test_data = dset.CIFAR10(root=opt.datadir, download=True, train=False, transform=transform)
    elif opt.dataset == "svhn":
        test_data = dset.SVHN(root=opt.datadir, download=True, split="test", transform=transform)
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
            if x.shape[1] == 3:
                # ----------------------------------------------- #
                # add in contour: gray(orig) - gray(Gaussian(orig))
                gray_orig = transforms.Grayscale(num_output_channels=1)(x)
                gaus_orig = transforms.GaussianBlur(kernel_size=7)(x)
                gray_blur = transforms.Grayscale(num_output_channels=1)(gaus_orig)
                contour = (gray_orig - gray_blur).clamp(0, 1)
                x = torch.cat((contour, x), dim=1).clone().detach()
            
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
    
    NLL, NLL_regret = [], []
    
    for i, (xi, _) in enumerate(test_queue):
        x = xi.expand(opt.repeat, -1, -1, -1).contiguous()
        
        # ----------------------------------------------- #
        # add in contour: gray(orig) - gray(Gaussian(orig))
        gray_orig = transforms.Grayscale(num_output_channels=1)(x)
        gaus_orig = transforms.GaussianBlur(kernel_size=7)(x)
        gray_blur = transforms.Grayscale(num_output_channels=1)(gaus_orig)
        contour = (gray_orig - gray_blur).clamp(0, 1)
        x = torch.cat((contour, x), dim=1).clone().detach()
        
        # compute the negative log-likelihood before optimizing q(z|x)
        NLL_loss_before = nll_helper(netE).detach().cpu().numpy()
        NLL = np.append(NLL, NLL_loss_before)
    
        # optimize wrt to the single sample
        xi = xi.to(device, non_blocking=True)
        
        # ----------------------------------------------- #
        # add in contour: gray(orig) - gray(Gaussian(orig))
        gray_orig = transforms.Grayscale(num_output_channels=1)(xi)
        gaus_orig = transforms.GaussianBlur(kernel_size=7)(xi)
        gray_blur = transforms.Grayscale(num_output_channels=1)(gaus_orig)
        contour = (gray_orig - gray_blur).clamp(0, 1)
        xi = torch.cat((contour, xi), dim=1).clone().detach()
        
        b = xi.size(0)
        netE_copy = copy.deepcopy(netE)
        netE_copy.eval()
        optimizer = optim.Adam(netE_copy.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=5e-5)
        target = (xi.view(-1)*255).to(torch.int64)
        
        for it in range(opt.num_iter):
            z, mu, logvar = netE_copy(xi)
            recon = netG(z)
            recon = recon.contiguous()
            recon = recon.view(-1, 256) # distribution over 256 classes
            
            recl = loss_fn(recon, target) # target labels
            recl = torch.sum(recl) / b
            kld = KL_div(mu,logvar)
            loss =  recl + kld.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # compute the negative log-likelihood after optimizing q(z|x)
        NLL_loss_after = nll_helper(netE_copy).detach().cpu().numpy()
        regret = NLL_loss_before - NLL_loss_after
        NLL_regret = np.append(NLL_regret, regret)
        
        print(f"Image {i:d} [before] {NLL_loss_before:.3f} [after] {NLL_loss_after:.3f} [diff] {regret:.3f}")
        if i >= 1000: # test for 5000 samples
            break
    np.save(os.path.join(opt.savedir, f"NLL.npy"), NLL)
    np.save(os.path.join(opt.savedir, f"regret.npy"), NLL_regret)
    

if __name__ == "__main__":
    main()