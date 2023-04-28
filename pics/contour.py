import torch

import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.io import read_image
import torch.utils.data
import torchvision.datasets as dset

import os

def main(name, kernel, sigma):
    os.makedirs(name, exist_ok=True)
    
    transform = transforms.Compose([transforms.Resize(32), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
    dataset = dset.MNIST(root="../data/", train=False, download=True, transform=transform)
    
    # x = read_image(name+".png")
    x = dataset[0][0]
    # x = x[:3] / 255
    # add in contour: gray(orig) - gray(Gaussian(orig))
    gray_orig = transforms.Grayscale(num_output_channels=1)(x)
    # save_image(gray_orig, f"cat/gray_orig_{kernel}_{sigma}.jpg")
    
    
    gaus_orig = transforms.GaussianBlur(kernel_size=kernel, sigma=sigma)(x)
    # save_image(gaus_orig, f"cat/blur_{kernel}_{sigma}.jpg")
    gray_blur = transforms.Grayscale(num_output_channels=1)(gaus_orig)
    save_image(gray_blur, f"{name}/gray_blur_{kernel}_{sigma}.jpg")
    contour = gray_orig - gray_blur
    contour = (contour+0.5).clamp(0, 1)
    save_image(contour, f"{name}/contour_{kernel}_{sigma}.jpg")
    

if __name__ == "__main__":
    name = "mnist"
    
    for kernel in [9, 51, 101]:
        for sigma in [3, 10, 20]:
            main(name, kernel, sigma)