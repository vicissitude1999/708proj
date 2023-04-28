# Out-of-Distrbution Detection

## Training
### Training with addition of contour
In our data augmentation implementation, contour information of an image is added as the 4th channel before passing it into the VAE for training. The contour implementation is added to `train.py`. Model parameters are included in the json files in the `/tools` folder.

After training, there will be a new folder containing saved model under corresponding dataset folder in the`/output` directory. This folder name will be used in the command line during testing.

Example:
```
python src/train.py tools/train_cifar10.json
```
### Training baseline with no contour
```
python src/train_orig.py tools/train_cifar10.json
```

## Testing 
### Computing Likelihood Regret.
Compute the Likelihood Regret and prepare for plotting. Specify the training directory (folder saved when training the model) and dataset name.

After running, there will be a new folder under the training directory, with name like `test-20230422-135210`. This will be used to plot the histogram and ROC curve.

Example with contour:
```
python src/compute_LR.py --train_dir output/cifar10/20230421-222638 --dataset cifar10
python src/compute_LR.py --train_dir output/cifar10/20230421-222638 --dataset svhn
```
Example baseline without contour:
```
python src/compute_LR_orig.py --train_dir output/cifar10/20230421-222638 --dataset cifar10
python src/compute_LR_orig.py --train_dir output/cifar10/20230421-222638 --dataset svhn
```

### Computing NLL-IC
Compute the Negative Log-Likelihood scaled with Input Complexiy and prepare for plotting.

Example with contour:
```
python src/compute_NLL_IC.py --train_dir output/cifar10/20230421-222638 --dataset cifar10
python src/compute_NLL_IC.py --train_dir output/cifar10/20230421-222638 --dataset svhn
```
Example baseline without contour:
```
python src/compute_NLL_IC_orig.py --train_dir output/cifar10/20230421-222638 --dataset cifar10
python src/compute_NLL_IC_orig.py --train_dir output/cifar10/20230421-222638 --dataset svhn
```

## Plot histograms and ROC curves

Specify two arguments, --indist_dir and --ood_dir, each containing the NLL and LR files.
After running, there will be a directory in the training directory named `metrics-{indist dataset}-{outdist dataset}`.

Example with LR as detection score:
```
python src/plt.py \
--indist_dir output/cifar10/20230421-222638/test-20230422-135210 \
--ood_dir output/cifar10/20230421-222638/test-20230422-141140
```
Example with NLL-IC as detection score:
```
python src/plt_NLL_IC.py \
--indist_dir output/cifar10/20230421-222638/test-20230423-001222
--ood_dir output/cifar10/20230421-222638/test-20230423-001539
```

## Pretrained models
Our trained `/output` directory can be found here:
https://drive.google.com/drive/folders/1ZobRhezwqcTntfZir2M1e88g3FhqZNf6?usp=share_link.
Download from the link and put the output/ directory in 708proj/.