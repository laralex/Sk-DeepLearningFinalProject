# Signal recovery from nonlinear distortion in optical communications
The final project of Deep Learning 2021 course at Skoltech, Russia.

Team members:
* Ilya Kuk
* Razan Dibo
* Mohammed Deifallah
* Sergei Gostilovich
* Alexey Larionov
* Stanislav Krikunov
* Alexander Blagodarnyi

Inspired by the papers:

1. [Advancing theoretical understanding and practical performance of signal processing for nonlinear optical communications through machine learning](https://www.nature.com/articles/s41467-020-17516-7).
2. [Fundamentals of Coherent Optical Fiber Communications ](https://www.osapublishing.org/jlt/abstract.cfm?URI=jlt-34-1-157)

### Brief repository overview
* 👉 [`train.py`](train.py) - entry point for training of models (see
  **Reproduce training** section)
* [`notebooks/training.ipynb`](notebooks/training.ipynb) - a quickstart Jupyter
  notebook for training

### Requirements
A GPU is recommended to perform the experiments. You can use [Google
Colab](colab.research.google.com) with Jupyter notebooks provided in
[`notebooks/`](notebooks/) folder

Main prerequisites are:

- [`pytorch`](http://pytorch.org/), [`pytorch-lightning`](https://www.pytorchlightning.ai/)
- `numpy`, `tensorboard`, `tqdm`


<!TODO:
For convenience, a ready to use conda [environment](environment.yml) is provided. 
To create a new python environment with all the required packages, you can run:
```shell
conda env create -f environment.yml
conda activate dyconv
```
!>

### Reproduce training
...
### Experimental results
...
