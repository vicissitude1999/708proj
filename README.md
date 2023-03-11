# Reproduce Likelihood regret
## Training

python src/train.py tools/train_fmnist.json

## Testing (computing LR)

Specify the training run and dataset name.

Example: python src/compute_LR.py --train_dir output/fmnist/20230309-213438 --dataset mnist

## Pretrained models

https://drive.google.com/drive/folders/1ZobRhezwqcTntfZir2M1e88g3FhqZNf6?usp=share_link

Download from the link and put the output/ directory in 708proj/. 