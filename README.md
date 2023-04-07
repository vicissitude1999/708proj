# Reproduce Likelihood regret
## Training

python src/train.py tools/train_fmnist.json

## Testing (computing LR)

Specify the training run and dataset name.

Example:
python src/compute_LR.py --train_dir output/fmnist/20230309-213438 --dataset fmnist

python src/compute_LR.py --train_dir output/fmnist/20230309-213438 --dataset mnist

python src/compute_LR.py --train_dir output/cifar10/20230309-213812 --dataset cifar10

python src/compute_LR.py --train_dir output/cifar10/20230309-213812 --dataset svhn


## Plot histograms and ROC curves

Specify two arguments, --indist_dir and --ood_dir, each containing the NLL and LR files.
It will create a directory in the training directory named "metrics-{indist dataset}-{outdist dataset}".

For example:
Trained on FMNIST, test on FMNIST and MNIST:

python src/plt.py \
--indist_dir output/fmnist/20230309-213438/test-20230406-193701 \
--ood_dir output/fmnist/20230309-213438/test-20230406-193706

Trained on Cifar10, test on Cifar10 and SVHN:

python src/plt.py \
--indist_dir output/cifar10/20230309-213812/test-20230406-113230 \
--ood_dir output/cifar10/20230309-213812/test-20230406-113234



## Pretrained models

https://drive.google.com/drive/folders/1ZobRhezwqcTntfZir2M1e88g3FhqZNf6?usp=share_link

Download from the link and put the output/ directory in 708proj/.