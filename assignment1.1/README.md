# Assignment 1.1: Convolutional Neural Networks (CNNs)

Assignment Author: Peyton Chandarana

## Objective:

## 0. Preliminaries:

This tutorial can be run on multiple different platforms. You can choose one of the following or another of your choice.

- Locally (a computer of your choice)
- Kaggle (website)
- Google Colab (website)

### Locally

Running the tutorial on a computer of your choice is completely fine and you may use the `requirements.txt` file provided to reproduce the Conda + pip environment that was used to develop this assignment. However, you must first have a form of Anaconda or Conda installed on your computer prior to starting the assignment.

**_NOTE: If your computer does not have an NVIDIA GPU, it may take a while to train/test the model. Use another option if you cannot train on a system of your own._**

To do this, I recommend using `miniforge` found here:

https://github.com/conda-forge/miniforge

Once you install Miniforge3 on your device use the following command to create the Conda environment with Python 3.12 installed:

```shell
mamba create -n assignment1 python=3.12
```

We only use conda to manage the Python versions here since PyTorch no longer supports the conda PyTorch repository.

**_NOTE: You may need to run the following command and restart your shell to initialize the `mamba` command! Using mamba versus conda has generally been faster in the past, but should be the same now._**

```shell
mamba init
```

or be more specific by adding the shell at the end of the command:

```shell
mamba init /bin/bash
```

If you are using a Mac or you are like me and prefer ZSH replace `/bin/bash` with `/bin/zsh`.

Once you have the Conda environment initialized, you can install the packages via pip using `pip install -r requirements.txt` to install all of the packages via pip.

Open the `main.ipynb` notebook in either VSCode (may require some extensions to be installed - see `vscode_extensions.txt`) or using a browser by launching the Jupyter Notebook server locally.

### Kaggle

We will be using Kaggle later so if you wish to use Kaggle for this assignment it should be familiar in subsequent assignments.

To use on Kaggle, you just have to create an account and then open the `main.ipynb` file on Kaggle. Keep in mind that you may get different results since the package versions found in the `requirements.txt` file may not exactly match the ones on Kaggle.

### Google Colab

Google Colab is another option to use for running this tutorial and is very similar to Kaggle. If you have a Google or Gmail account you can easily open a Google Colab notebook and import the `main.ipynb` into it.

Similarly to Kaggle, keep in mind that you may get different results since the package versions found in the `requirements.txt` file may not exactly match the ones on Google Colab.

## 1. Train LeNet-5

Fully run the tutorial by starting up a notebook kernel on your local machine, Kaggle, or Google Colab.

This should be as simple as pressing the "Run all" or similar play button.

**_NOTE: The code in `main.ipynb` is very long and verbose. Not all of this code is necessary, but it provides a good introductions of how to create models, compare them, and analyze the results._**

## 2. Deliverables

There are several questions in throughout the `main.ipynb`. Answer them to the best of your knowledge and submit them to blackboard.

## Note on Training Issues During Live Demo in Class

The reason we were not getting poor accuracy and high loss during class was due to how we divided the dataset by 255.0 to normalize the grayscale images into values between 0.0 and 1.0. The bug was in these lines of code during the demo:

```
# DO NOT DO THIS
train_data.data = ...
test_data.data = ...
```

If you use `plt.show(...)` after performing these lines, you will see that the digits are unrecognizable because the data was changed during this wrong step.

Changing this to be performed as part of a custom transform fixed the issue:

```python
# Transform function
transform = tv.transforms.Compose([
    # Convert to tensor
    tv.transforms.ToTensor(),
    # Divide by 255 to get values between 0 and 1
    tv.transforms.Lambda(lambda x: x.float() / 255.0)
])

# Import MNIST dataset
train_data = tv.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_data = tv.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
```
