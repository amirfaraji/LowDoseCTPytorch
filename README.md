# LowDoseCTPytorch
LoDoPaB-CT Grand Challenge Code


## The Challenge and Task
The task of this challenge is to reconstruct CT images of the human lung from (simulated) low photon count measurements. For evaluation, the PSNR and SSIM values are computed w.r.t. the images that were used as ground truth.

## Setup

##### Create environment
```python3 -m venv lowdosect```

##### Activate environment

```source lowdosect/bin/activate```

##### Install packages
1. Install pytorch first

```pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html```

2. Use requirements.txt for other requirements

```pip install -r src/requirements.txt```



3. Install Astra toolbox (recommended for RayTransform)

**If astra toolbox is not installed please change line 13 in RayTransform.py from:**

```impl = 'astra_cpu',```

to:

```impl = 'skimage',```

Everything under the /src/libs folder is external libraries I wasn't able to install through pip. This includes LoDoPaB-CT challenge utilities (https://github.com/jleuschn/lodopab_challenge) and astra toolbox (https://github.com/astra-toolbox/astra-toolbox) I used the code from the master branch found on their respective github pages.

The astra toolbox was a little tricky to build. Using conda would have been better, but I was using venv. To install you will need to do the follow the guide found on their guide github page for building from source. 

## Running code

#### Training 
To run the training, inference and submission code you will need to have the data save in src/data/ folder. The folder should contain the unzipped data with the name ground_truth_{phase}, observation_{phase} and observation_challenge.
```
python src/train.py
```
```
usage: train.py [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--lr LR] [--workers WORKERS] [--patience PATIENCE] [--path PATH] [--load-model]

Training Low Dose CT Scans

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        input batch size for training (default: 4)
  --epochs EPOCHS       number of epochs to train (default: 8)
  --lr LR               learning rate (default: 1e-5)
  --workers WORKERS     number of workers for data loader (default: 4)
  --patience PATIENCE   random seed (default: 15)
  --path PATH           Path for loading Model
  --load-model          For loading the Model

```
**Note: There is a patience parameter and a scheduler in the code but it is unused.** 
#### Inference

```
python src/inference.py
```

``` 
usage: inference.py [-h] [--test-batch-size B] [--workers WORKERS] [--path PATH]

Inference for Low Dose CT Scans

optional arguments:
  -h, --help           show this help message and exit
  --test-batch-size B  input batch size for test (default: 1)
  --workers WORKERS    number of workers for data loader (default: 12)
  --path PATH          Path for loading Model 
```
#### Submission

```
python src/submission.py
```

## Model

The final architecture used for the challenge was a UNet++ implemented in PyTorch. The model can be found under src/model/Models.py under ```BiggerUnetPlusPlus```. Paper can be found here: https://arxiv.org/pdf/1807.10165.pdf

Filtered Backprojection was used on the observed data and no other preprocessing techniques were used. Vertical flip, horizontal flip and random crop-resize was used during training.

## Hyperparamters

The hyperparameters used are the following:
- batch size = 4
- epochs = 8
- learning rate = 1e-5
- num_workers = 4
- Image size = (362,362) (No changes to the dimensions)

The lose function was a mixed MSE and SSIM described by this ```(1-alpha)*ssim_loss + alpha*mse ``` where alpha was 0.35.

The optimizer was RMSprop with weight_decay=1e-8 and momentum=0.9

## Hardware
Pytorch API was employed to implement the architecture. Training  was  done on Ubuntu with a GeForce  GTX  2070Super GPU with 8  GB memory and 16 GB of RAM.


## Acknowledgement

Deep Inversion Validation (dival) Library (https://github.com/jleuschn/dival) along with LoDoPaB-CT challenge utilities (https://github.com/jleuschn/lodopab_challenge) were used for submission. 
I would like to thank the following authors of the libraries for its use.

    Daniel Otero Baguer otero@math.uni-bremen.de
    Mateus Baltazar
    David Erzmann erzmann@uni-bremen.de
    Johannes Leuschner jleuschn@uni-bremen.de
    Maximilian Schmidt maximilian.schmidt@uni-bremen.de
