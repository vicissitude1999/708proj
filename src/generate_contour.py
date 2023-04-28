import sys
import json
from addict import Dict

import torch.utils.data
from torchvision.utils import save_image
import torchvision.datasets as dset
import torchvision.transforms as transforms


def parse_args():
    config = sys.argv[1] # tools/xx.json
    with open(config) as f:
        opt = json.load(f)
    opt = Dict(opt)
    return opt


device = "cuda:0"
opt = parse_args()
data_api = dset.CIFAR10
train_data = data_api(root=opt.datadir, train=True, download=True, transform=transforms.Compose([
                                  transforms.Resize((opt.imageSize)),
                                  transforms.ToTensor()]))
train_queue = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=opt.workers)
for step, (x, y) in enumerate(train_queue):
    x = x.to(device, non_blocking=True)
    
    save_image(x, "orig.png")
    # add in contour: gray(orig) - gray(Gaussian(orig))
    gray_orig = transforms.Grayscale(num_output_channels=1)(x)
    save_image(gray_orig, "gray_orig.png")
    gaus_orig = transforms.GaussianBlur(kernel_size=9, sigma=3)(x)
    save_image(gaus_orig, "blur.png")
    gray_blur = transforms.Grayscale(num_output_channels=1)(gaus_orig)
    save_image(gray_blur, "gray_blur.png")
    contour = gray_orig - gray_blur
    contour = (contour+0.5).clamp(0, 1)
    save_image(contour, "contour.png")
    break