# Code for "Out-of-distribution Detection by Cross-class Vicinity Distribution of In-distribution Data"

## requirement
* Python 3.7
* Pytorch 1.1
* scikit-learn
* tqdm
* pandas
* scipy

## Datasets
### In-distribution Datasets
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [SVHN](http://ufldl.stanford.edu/housenumbers/)

Our codes will download the two in-distribution datasets automatically.

### Out-of-Distribtion Datasets
The test out-of-distribution datasets are provided by [CSI](https://github.com/alinlab/CSI)

Each out-of-distribution dataset should be put in the corresponding subdir in [./data_LCVD](./data_LCVD)

## Train and Test
Run the script [demo.sh](./code_LCVD/demo.sh). 
