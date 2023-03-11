import os
import shutil

import numpy as np
import torch


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def create_exp_dir(path, scripts_to_save=None):
    os.makedirs(path, exist_ok=True)
    if scripts_to_save:
        os.makedirs(os.path.join(path, "scripts"), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(state, is_best, save):
    os.makedirs(save, exist_ok=True)
    filename = os.path.join(save, "current.ckpt")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, "best.ckpt")
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))