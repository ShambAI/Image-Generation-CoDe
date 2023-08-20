
# Generating Realistic Underwater Images Using Contrastive Learning

In this project, we investigate the use of contrastive learning and generative adversarial network to build an end-to-end neural network for generating realistic underwater images from a set of uniform lighting synthetic inputs conditioned on real underwater images. We use the contrastive loss to preserve the content of the generated images and modify our model architecture to account for 4-channel RGBD input. We performed inference on 190 uniform lighting images from the Vision Autonomous Robots for Ocean Sustainability (VAROS) synthetic underwater data environment and computed the Fréchet inception distance (FID) to evaluate the realism of the generated images. Two key quantitative metrics were used to evaluate our model

+ Fréchet inception distance (FID)
+ Structural Similarity Index Metric (SSIM)

Feature Space of Both Classes. This is also another qualitative metric. In this metric, we
simply perform a principal component analysis (PCA) on both the uniform lighting images
and underwater images from the VAROS dataset on a 2D feature space to see the space both
categories of data occupied and if at all both classes are linearly separable

## Installation
First install the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
sudo apt install python3-pip
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the modules in the requirements.txt file. 

```bash
pip install -r requirements.txt
```

This should be enough to get started. However, if there is an issue with `pytorch` not finding cuda, try this script below.

```bash
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
The project utilizes [weight and biases](https://docs.wandb.ai/quickstart) to visualize metrics such as loss and accuracy during training. The installation is a part of the `requirements.txt` file, hence there is no need to reinstall. Run the below command line snippet, put in [your API key](https://wandb.ai/authorize) when prompted, and you'll see the new run appear in W&B.
```bash
wandb login
```

## Project File Description


#### Executable files:
+ train.py: This executable file takes in several parameters including but not limited to the real and synthetic data, train batch size, number of gpus and output number of channels. This file is used for training our cut model.

+ test.py: The core function of this file is to perform inference using the latest model from the training phase. It takes in the test data path and train model path

#### Output files:
+ The execution of the `train.py` produces some useful output that can be found in the output terminal. Also, the trained models after every specified epoch are stored in the checkpoint folder

+ The `test.py` produces an output file in the form of an image that displays the realistic underwater images as shown below.

![Alt text](./data/Results.jpg?raw=true "Title")

## Execution and Usage
+ #### Self-supervised pretext task:
The pretext tasks are trained separately using VISSL or Autoencoder the weights are transferred for downstream fully supervised classification

Where the pretext task can be either one of the following:
> SimCLR or Inpainting 


+ #### Fully supervised downstream fine-tuning for anomaly detection:
Executing the `resnet_classification.py` uses transfer learning to perform a direct classification task on a small subset of labeled data using the script highlighted below.

```bash
python3 resnet_classification.py
```

## Company
[Alteia ](https://alteia.com/software/) [NTNU ](https://www.ntnu.edu/)
