import os
import argparse

import json
from addict import Dict
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

def plot_roc_curve(fpr, tpr, output_dir, label):
    plt.plot(fpr, tpr, color='orange', label=label)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve.jpg"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indist_dir", type=str, help="directory of indistribution outputs")
    parser.add_argument("--ood_dir", type=str, help="directory of out distribution outputs")
    
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    with open(os.path.join(args.indist_dir, "opt.json"), "r") as f:
        args_indist = Dict(json.load(f))
    with open(os.path.join(args.ood_dir, "opt.json"), "r") as f:
        args_ood = Dict(json.load(f))
    
    train_dir = "/".join(args.indist_dir.split("/")[0:-1])
    output_dir = os.path.join(train_dir, f"metrics-{args_indist.dataset}-{args_ood.dataset}")
    os.makedirs(output_dir, exist_ok=True)
    
    ################ Likelihood
    ll_indist = -np.load(os.path.join(args.indist_dir, "NLL.npy"))
    ll_ood = -np.load(os.path.join(args.ood_dir, "NLL.npy"))
    min_ll = min(ll_indist.min(), ll_ood.min())
    max_ll = max(ll_indist.max(), ll_ood.max())
    bins_ll = np.linspace(min_ll, max_ll, 50)

    plt.hist(ll_indist, bins_ll, alpha=0.5, label=f"In-distribution:{args_indist.dataset}")
    plt.hist(ll_ood, bins_ll, alpha=0.5, label=f"OOD:{args_ood.dataset}")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_dir, f"nll_indist_{args_indist.dataset}_ood_{args_ood.dataset}.jpg"))
    plt.close()

    ################ Regret
    regret_indist = np.load(os.path.join(args.indist_dir, "regret.npy"))
    regret_ood = np.load(os.path.join(args.ood_dir, "regret.npy"))
    min_regret = min(regret_indist.min(), regret_ood.min())
    max_regret = max(regret_indist.max(), regret_ood.max())
    bins_regret = np.linspace(min_regret, max_regret, 50)

    plt.hist(regret_indist, bins_regret, alpha=0.5, label=f"In-distribution:{args_indist.dataset}")
    plt.hist(regret_ood, bins_regret, alpha=0.5, label=f"OOD:{args_ood.dataset}")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_dir, f"regret_indist_{args_indist.dataset}_ood_{args_ood.dataset}.jpg"))
    plt.close()
    
    
    ################ AUC-ROC
    regret_indist = np.load(os.path.join(args.indist_dir, "regret.npy"))
    regret_ood = np.load(os.path.join(args.ood_dir, "regret.npy"))
    
    combined = np.concatenate((regret_indist, regret_ood))
    label_1 = np.ones(len(regret_indist))
    label_2 = np.zeros(len(regret_ood))
    label = np.concatenate((label_1, label_2))

    fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=0)

    plot_roc_curve(fpr, tpr, output_dir, label="LR")

    rocauc = metrics.auc(fpr, tpr, )
    print('AUC for likelihood regret is: ', rocauc)



    nll_cifar = np.load(os.path.join(args.indist_dir, "NLL.npy"))
    nll_svhn = np.load(os.path.join(args.ood_dir, "NLL.npy"))

    combined = np.concatenate((nll_cifar, nll_svhn))
    label_1 = np.ones(len(nll_cifar))
    label_2 = np.zeros(len(nll_svhn))
    label = np.concatenate((label_1, label_2))

    fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=0)

    plot_roc_curve(fpr, tpr, output_dir, label="NLL")

    rocauc = metrics.auc(fpr, tpr)
    print('AUC for nll is: ', rocauc)



if __name__ == "__main__":
    main()