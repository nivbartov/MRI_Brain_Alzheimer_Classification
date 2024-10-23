<h1 align="center">MRI Brain Alzheimer Classification Under Adversarial Attacks</h1>

<h1 align="center">
  <img src="assets/icons/main_image_readme.jpg" height="300">
</h1>

<div align="center">
    <p><strong>This project is a part of the EE046211 Deep Learning course at the Technion.</strong></p>
</div>


<h4 align="center">
    Dor Lerman:
    <a href="https://www.linkedin.com/in/..."><img src="assets/icons/Linkedin_icon_readme.png" width="30" height="30"/></a>
    <a href="https://github.com/dorlerman"><img src="assets/icons/GitHub_icon_readme.png" width="30" height="30"/></a>
</h4>

<h4 align="center">
    Niv Bar-Tov:
    <a href="https://www.linkedin.com/in/niv-bar-tov"><img src="assets/icons/Linkedin_icon_readme.png" width="30" height="30"/></a>
    <a href="https://github.com/nivbartov"><img src="assets/icons/GitHub_icon_readme.png" width="30" height="30"/></a>
</h4>


<h4 align="center">
    <a href="https://colab.research.google.com/...."><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <a href="https://www.youtube.com/watch?v=YOUR_VIDEO_ID" target="_blank"><img src="https://img.shields.io/badge/Watch%20Video-YouTube-red?logo=youtube" alt="Watch Video on YouTube"/></a>
</h4>



## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Files In The Repository](#files-in-the-repository)
- [Environment Installation](#environment-installation)
- [Dataset](#dataset)
- [Trained Models](#trained-models)
- [Results](#results)
- [Usage](#usage)
- [Sources and References](#sources-and-references)
- [Citation](#citation)
- [License](#license)

## Project Overview
In the realm of cybersecurity, significant efforts are focused on safeguarding computing systems from digital threats, which are increasingly prevalent today. In the context of machine learning, adversarial attacks pose a unique challenge by intentionally manipulating input data to deceive models. This altered data, crafted to appear legitimate, causes models to make incorrect predictions. Medical imaging systems, like many others, are particularly susceptible to such adversarial attacks, underscoring the critical need to strengthen the resilience of these models against such vulnerabilities.

In this project, we aim to design a robust model for detecting and classifying Alzheimer disease using MRI brain images. The model simulates a radiologist's diagnostic process by classifying images into four severity levels. We evaluate and compare several well-known unsupervised pre-trained models for classification tasks. Then, we train and evaluate these models under adversarial attacks, which can significantly reduce model's performance. By combining the models, we aim to create an ensemble, a unified and robust model that maximizes resilience against adversarial attacks while maintaining high classification performance. 

The project includes the following steps:

1. **Transfer Learning:** We used transfer learning to fine-tune and extract features from three well-known unsupervised pre-trained models to perform well on our specific task: DINOv2, ResNet34, and EfficientNet-B0. 

2. **Adversarial Attacks Implementation:** We performed two adversarial attacks on each one of the models: Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD). 

3. **Adversarial Training:** To enhance model robustness, we trained these models with adversarial input, focusing particularly on the PGD attack. 

4. **Ensemble Models:** Finally, we combined these three models using a voting approach to create a robust model, without affecting performance. 

The project is implemented in Python using the PyTorch framework, which allows us to build and train the models efficiently throughout these steps.

## Repository Structure

| Directory Name | Content |
|----------------|---------|
| `assets` | Contains images for each model, including confusion matrices, loss curves, and accuracy curves. |
| `dataset` | Contains two sub-directories: `raw_dataset` with the original raw data, and `dataset_variables` with processed dataset splitted into train, validation and test sets. |
| `env` | Contains the project environment configuration files for setting up the required dependencies. |
| `models` | Contains all models training notebooks. |
| `utils` | Contains utility scripts. |

## Files In The Repository

| File Name | Description |
|-----------|-------------|
| `dataset/prepare_dataset.ipynb` | Notebook used to split the raw dataset into train, validation and test sets and applies a resize transformation of 224x224 pixels. |
| `dataset/dataset_variables/*.pt` | Processed dataset files: `train_set.pt`, `validation_set.pt`, and `test_set.pt` (created by `prepare_dataset.ipynb`). |
| `env/requirements.txt` | List of required Python packages for setting up the environment. |
| `env/project_env.yaml` | Environment configuration file. |
| `utils/optuna_search.py` | Script for performing hyperparameters search using Optuna. It allows customization of epochs, trials and hyperparameters. |
| `utils/Grad_cam.py` | Script for generating Grad-CAM heatmaps. User must specify required parameters as per function definitions. |
| `utils/utils_funcs.py` | Contains general utility functions such as saving models, loading images, displaying graphs, and training. |
| `models/def_models.py` | Definition of class objects used for the trained models. |
| `models/*_model.ipynb` | Model-specific notebook (e.g. `resnet_model.ipynb`). Used for data loading, training, evaluation and results generation: accuracy, confusion matrix, loss curve and accuracy curve. |
| `models/*_model_atk.ipynb` | Adversarial training model-specific notebook (e.g. `resnet_model_atk.ipynb`). Loads a pre-trained model, applies adversarial attacks, and trains the model under these attacks. |


## Environment Installation

#### General Prerequisites

| Library           | Version           |
|-------------------|-------------------|
| `Python`          | `3.10`            |
| `torch`           | `>= 1.4.0`        |
| `matplotlib`      | `>= 3.7.1`        |
| `numpy`           | `>= 1.24.3`       |
| `opencv`          | `>= 4.5.0`        |
| `pandas`          | `>= 1.5.0`        |
| `tqdm`            | `>= 4.65.0`       |
| `scipy`           | `>= 1.8.1`        |
| `seaborn`         | `>= 0.11.2`       |
| `plotly`          | `>= 5.7.0`        |
| `notebook`        | `>= 6.5.4`        |
| `ipywidgets`      | `>= 7.6.0`        |
| `torchmetrics`    | `>= 0.10.0`       |
| `optuna`          | `>= 2.10.0`       |
| `fvcore`          | `>= 0.1.5`        |
| `iopath`          | `>= 0.1.9`        |
| `submitit`        | `>= 1.3.0`        |
| `kornia`          | `>= 0.6.0`        |
| `prettytable`     | `>= 2.4.0`        |
| `pickleshare`     | `>= 0.7.5`        |
| `torchcam`        | `>= 0.1.2`        |
| `torchattacks`    | `>= 0.2.0`        |

#### DINOv2 Specific Requirements

| Library           | Version           |
|-------------------|-------------------|
| `torch`           | `== 2.0.0`        |
| `torchvision`     | `== 0.15.0`       |
| `omegaconf`       | `Latest`          |
| `torchmetrics`    | `== 0.10.3`       |
| `fvcore`          | `Latest`          |
| `iopath`          | `Latest`          |
| `submitit`        | `Latest`          |
| `xformers`        | `== 0.0.18`       |
| `cuml-cu11`       | `Latest`          |


To set up the required dependencies, please follow one of the options below:

##### 1. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) (Recommended)
Clone this repository and then create and activate the conda environment (`env/project_env.yaml`) using the following commands:

```
conda env create -f env/project_env.yaml
conda activate project_env
```
##### 2. Pip Install

Clone this repository and then use the provided `env/requirements.txt` file to install the required dependencies:

```
pip install -r env/requirements.txt
```

## Dataset

We used a pre-processed dataset of 11,519 axial MRI brain images: 6,400 images from real patients and 5,119 synthetic images that were developed to rectify the class imbalance of the original dataset. The images are classified into four categories: "Non Demented", "Very Mild Demented", "Mild Demented", and "Moderate Demented". Each category had 100, 70, 28, and 2 patients, respectively, and each patient's brain was sliced into 32 horizontal axial MRIs. The images have a resolution of 128x128 pixels and are in the “.jpg” format. All images have been pre-processed to remove the skull.

The dataset was split according to the train-validation-test methodology: the train set contains 8,192 real and synthetic images, the validation set contains 2,048 real and synthetic images and the test set contains 1,279 real images only. We resized the images into 224x224 pixels to match the input size required for the pre-trained models.

## Trained Models

| Model Type                               | Google Drive Link                                                                                | Optuna Params                                                                                |
|------------------------------------------|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| DINOv2                                   | [Download .pth file](https://drive.google.com/file/d/1jABpNVpMTrehBhL6ilm2e0gTx2C7fQyd/view?usp=drive_link)      | [Download JSON file](https://drive.google.com/file/d/1EyuLRh44TgUEpMn5ErN77XdZEmFZ8MNd/view?usp=drive_link) |
| DINOv2 with adversarial attacks          | [Download .pth file](....)       | [Download JSON file](https://drive.google.com/file/d/1EyuLRh44TgUEpMn5ErN77XdZEmFZ8MNd/view?usp=drive_link) | 
| Resnet34                                 | [Download .pth file](.....)       | [Download JSON file](....) | 
| Resnet34 with adversarial attacks        | [Download .pth file](....)       | [Download JSON file](....) | 
| Efficientnet-B0                          | [Download .pth file](https://drive.google.com/file/d/1SuudU3uv2FCtJiD1XKdfpAPaMnk70Qhw/view?usp=drive_link)       | [Download JSON file](https://drive.google.com/file/d/1Bha09qvH3MlP112wm-j9j_NqijC5nPFL/view?usp=drive_link) | 
| Efficientnet-B0 with adversarial attacks | [Download .pth file](.....)        | [Download JSON file](https://drive.google.com/file/d/1Bha09qvH3MlP112wm-j9j_NqijC5nPFL/view?usp=drive_link) | 

## Results

## Usage

In this repository, you have the following usage options:

#### Train a specific model

1. copy one of the models notebook **`dino_v2_model.ipynb`** , **`efficienetnet_model.ipynb`** ,**`resnet_model.ipynb`**. rename the new notebook to be **`*_model.ipynb`** where * is the name of your backbone.
2. follow up the notebooks structure:
  2.1. define the models architecture accordinglly. add the architecture to the file **`models/def_models.py`**
  2.2. If you are interested in transfer learning, set the **`requires_grad = False`** for layers you want to freeze. Otherwise, **`requires_grad = True`** for layers you want to fine tune.
  2.3. set **`load_existing_hyperparams = False`** if you want to perform the optuna search for the specific architecture and constraints of the model. define the number of epochs and trials for the search process as input parameters to  the function.
  2.4. define the number of epochs for the training.
  2.5. define the augmentations if needed.
  2.6. in the training process you have 2 options:
    *  Train from scratch. For this option , set the **`load_existing_params = False`**. 
    * reload an exisiting model. you can reload an existing model  that was previously trained and choose to train it or not.  Loading an exisiting model set **`load_existing_params =  True`**. for training a loaded model, set the **`train_loaded_model = True`** in addition.
  2.7. After the training process, the confusion matrix, loss curve and accuracy curve are saved under the `assets/*` when  `*` is the models name.

#### Adversarial attacks: training and attacking
**Note:** This option can be performed only if you have a **pretrained model** with a defined architecture under the **`models/def_models.py`** file.

Here are the common steps for both options:
1. copy one of the models notebook **`dino_v2_model_atk.ipynb`** , **`efficienetnet_model_atk.ipynb`** ,**`resnet_model_atk.ipynb`**. rename the new notebook to be **`*_model_atk.ipynb`** where * is the name of your backbone.
2. set the models definition from the training process under no attacks.
3. set **`load_existing_hyperparams = True`** if you want to use the previous optuna hyper-parameters that you found in the first training process under no attacks. This is the recommended approach. Otherwise you can set **`load_existing_hyperparams = False`** and make a new optuna search.
4. Load the pretrained model. set the **`pretrained_model_path`** parameter to the path of your pretrained model.

##### Perform attacks

* **FGSM attack** -  define the `Epsilon` value you want to perform the attack with. It is recommended to set a small number of `Epsilon` if you are limited with resources.
plot the adversarial examples using the `utils_funcs.plot_adversarial_examples` function. 
here is how:
`utils_funcs.plot_adversarial_examples(epsilons, examples, attack_name='fgsm', parameter_type = 'Epsilon')`.
This function will show the correct labels and the adversarial classification of the model for each `Epsilon`.

* **PGD attack** -  first, define the `Epsilon` and `num_iter` values you want to perform the attack with.Then, define the `Alpha` values.They can be smaller than `Epsilon/num_iter`(the `Epsilon` defines the radious for petubations) only.
plot the adversarial examples using the `utils_funcs.plot_adversarial_examples` function. 
here is how:
`utils_funcs.plot_adversarial_examples(alphas, examples , attack_name='pgd', parameter_type = 'Alpha')`.
This function will show the correct labels and the adversarial classification of the model for each `Alpha`.


##### Adversarial attacks training

1. define the following parameters : `Epsilon`, `num_iter`, `Alpha`(see contraints under [PGD attack](#pgd-attack)), `adv_weight`, `num_epochs`.
2. define the number of epochs for the training.
3. define the augmentations if needed.
4. in the training process you have 2 options:
    *  Train from scratch. For this option , set the  **`load_existing_params = False`**. 
    * reload an exisiting model. you can reload an  existingmodel  that was previously trained and choose to   train itor not.  Loading an exisiting model   set**`load_existing_params = True`**. for training a   loadedmodel, set the **`train_loaded_model = True`**  inaddition.
5. After the training process, the confusion matrix, loss curve and accuracy curve are saved under the `assets/*` when  `*` is the models name.

## Sources and References

#### Sources

[1] Luke Chugh. (2021). Best Alzheimer MRI dataset. Kaggle dataset. [https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy)

[2] Facebook Research. (2023). DINOv2. GitHub repository. [https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

[3] Gil, J. (2020). PyTorch Grad-CAM. GitHub repository. [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

[4] Optuna. (2023). Optuna: A hyperparameter optimization framework. GitHub repository. [https://github.com/optuna/optuna](https://github.com/optuna/optuna)

[5] Hoki. (2020). Torchattack: PyTorch adversarial attack library. GitHub repository. [https://github.com/Harry24k/torchattacks](https://github.com/Harry24k/torchattacks)


#### References

[1] Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Understanding adversarial attacks on deep learning based medical image analysis systems. [arXiv:1907.10456](https://arxiv.org/abs/1907.10456)

[2] Zhang, Y., & Yu, L. (2023). Adversarial attacks on foundational vision models. [arXiv:2308.14597](https://arxiv.org/abs/2308.14597)

[3] Chen, X., Zhang, H., & Li, Y. (2022). Exploring adversarial attacks and defenses in vision transformers trained with DINO. [arXiv:2206.06761](https://arxiv.org/abs/2206.06761)

[4] Xie, L., & Wang, Z. (2023). DINOv2: Learning robust visual features without supervision. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)


[5] Tajbakhsh, N., Shin, J., Gurudu, S. R., & Hurst, R. T. (2022). What makes transfer learning work for medical images: Feature reuse & other factors. [arXiv:2203.01825](https://arxiv.org/abs/2203.01825)

[6] Zhang, H., Li, Y., & Chen, X. (2023). Adversarial Attack and Defense for Medical Image Analysis: Methods and Applications. [arXiv:2308.14597](https://arxiv.org/abs/2308.14597)

[7] Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)

## Citation
```
@misc{jacobgilpytorchcam,
  title={PyTorch library for CAM methods},
  author={Jacob Gildenblat and contributors},
  year={2021},
  publisher={GitHub},
  howpublished={\url{https://github.com/jacobgil/pytorch-grad-cam}},
}
```

```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```

```
@inproceedings{akiba2019optuna,
  title={{O}ptuna: A Next-Generation Hyperparameter Optimization Framework},
  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle={The 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2623--2631},
  year={2019}
}
```



## License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.
