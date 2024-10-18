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
    <a href="https://github.com/..."><img src="assets/icons/GitHub_icon_readme.png" width="30" height="30"/></a>
</h4>

<h4 align="center">
    Niv Bar-Tov:
    <a href="https://www.linkedin.com/in/..."><img src="assets/icons/Linkedin_icon_readme.png" width="30" height="30"/></a>
    <a href="https://github.com/..."><img src="assets/icons/GitHub_icon_readme.png" width="30" height="30"/></a>
</h4>


<h4 align="center">
    <a href="https://colab.research.google.com/...."><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <a href="https://www.youtube.com/watch?v=YOUR_VIDEO_ID" target="_blank"><img src="https://img.shields.io/badge/Watch%20Video-YouTube-red?logo=youtube" alt="Watch Video on YouTube"/></a>
</h4>



## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#Project-Structure)
- [Files In The Repository](#files-in-the-repository)
- [Installation](#installation)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [Sources and References](#sources-and-references)
- [Citation](#citation)
- [License](#license)

## Project Overview
In the cyber security world, efforts are made to protect computing systems from digital attacks which are an emerging thread nowadays. In machine learning, attackers employ Adversarial machine learning, a method designed to trick models using decieving data. This deceptive data is fed into the models as input, causing classifiers to make incorrect classifications.
We saw that in [reference [1]](https://arxiv.org/abs/1907.10456), medical images are vulnerable to adversarial attacks.
Acknowledging this vulnerability emphasizes the importance of enhancing the model's resilience.

In this project we try to make a robust model while achiving the two main goals :
1. defend against adverserial attacks in medical images , particularly in MRI alzhiemer classification. Meaning, whenever the model is under attack, it will still generalize and perform well.
2. Generalize well in the main classification task of the severity of the alziehmer disease.

In order to achieve our goals, we followed these steps:

1. **Model Fine-Tuning**: We fine-tuned three well-known models: DINOv2, ResNet34, and EfficientNet B0.

2. **Adversarial Attack Implementation**: We performed adversarial attacks using the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), which were found to be effective attacks based on the findings in [reference  [6]](https://arxiv.org/abs/2303.14133).

3. **Vulnerability Assessment**: After fine-tuning, we observed that the models were vulnerable to adversarial attacks. 

4. **Adversarial Training**: To enhance model robustness, we conducted adversarial training, focusing particularly on the PGD attack over the FGSM attack, as discussed in [reference [7]](https://arxiv.org/abs/2303.14133).

5. **Model Ensemble**: Finally, we combined the three models using a voting approach to create a more resilient ensemble.

The project is implemented in Python using the PyTorch framework, which allows us to efficiently build and train our models throughout these steps.

## Project Structure

| Directory Name | Content |
|----------------|---------|
| `assets` | Contains images for each model, including confusion matrices, loss curves, and accuracy curves. These curves display both training and validation data. |
| `checkpoints` | Includes the Optuna best parameters directory and all trained model checkpoints. This directory is organized by model type and includes both standard and adversarially trained versions. |
| `dataset` | Contains two subdirectories: `raw_dataset` with the original data, and `dataset_variables` with preprocessed datasets split into training, validation, and test sets. |
| `env` | Holds the project environment configuration file (`project_env.yaml`) and requirements files for setting up dependencies with different installation methods. |
| `models` | Contains all models organized by backbone type. Each model has a version with adversarial training (suffix `_atk`). Also includes a `results.ipynb` notebook that compiles all results. |
| `utils` | Houses utility scripts, including `optuna_search.py` for hyperparameter optimization, `utils_funcs.py` with general helper functions, and `grad_cam.py` for generating Grad-CAM visualizations. |

## Files In The Repository

| File Name | Description |
|-----------|-------------|
| `dataset/dataset_variables/*_set.pt` | Contains saved dataset splits: `train_set.pt`, `validation_set.pt`, and `test_set.pt`, created by `prepare_dataset.ipynb`. |
| `dataset/prepare_dataset.ipynb` | Notebook to split the raw dataset in `raw_dataset` into training, validation, and test sets. Allows customization of the split ratio and applies a 224x224 resizing transformation. |
| `env/project_env.yaml` | Environment configuration file for setting up the project. |
| `env/requirements.txt` | List of required Python packages for setting up the project. |
| `utils/optuna_search.py` | Script for performing hyperparameter search using Optuna. Allows customization of epochs, trials, and hyperparameters. New models can be added by following existing initialization patterns. |
| `utils/Grad_cam.py` | Script for generating Grad-CAM heatmaps. Requires a model with a convolutional layer. Users must specify parameters as per function definitions. |
| `utils/utils_funcs.py` | Contains general utility functions for tasks such as saving models, loading images, displaying graphs, and training. Includes specific functions for adversarial training. |
| `models/def_models.py` | In this file the models are defined as a class object. You can find the models we defined and used and you can also add your own. |
| `models/*_model.ipynb` | Model-specific notebook (e.g., `resnet_model.ipynb`). Covers the complete process: data loading, training, saving, and evaluation. Generates confusion matrices and plots loss/accuracy curves. |
| `models/*_model_atk.ipynb` | Adversarial training notebook for each model (e.g., `resnet_model_atk.ipynb`). Loads a pretrained model, applies adversarial attacks, and trains the model on these attacks. |


## Installation

This code works with any OS — Linux or Windows. To set up all the required dependencies, please follow one of the instructions below:

##### 1. Conda

[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) (Recommended) - Clone this repository and then create and activate the **`env/project_env.yaml`** conda environment using the provided environment definition:

```
conda env create -f env/project_env.yaml
conda activate project_env
```

##### 2. Pip Install

Clone this repository and then use the provided **`env/requirements.txt`** to install the dependencies.

```
pip install -r env/requirements.txt
```

#### General Prerequisites

| Library           | Version           | Why?                                                       |
|-------------------|-------------------|-----------------------------------------------------------|
| `Python`          | `3.10`            | Required Python version for the project                   |
| `torch`           | `>= 1.4.0`        | Core deep learning library for building models             |
| `matplotlib`      | `>= 3.7.1`        | For plotting graphs and visualizations                     |
| `numpy`           | `>= 1.24.3`       | Essential for numerical operations and data manipulation    |
| `opencv`          | `>= 4.5.0`        | For image processing tasks                                 |
| `pandas`          | `>= 1.5.0`        | For data manipulation and analysis                         |
| `tqdm`            | `>= 4.65.0`       | For progress bar visualization                             |
| `scipy`           | `>= 1.8.1`        | For scientific computing and technical calculations        |
| `seaborn`         | `>= 0.11.2`       | For statistical data visualization                         |
| `plotly`          | `>= 5.7.0`        | For interactive visualizations                             |
| `notebook`        | `>= 6.5.4`        | To run Jupyter Notebooks                                   |
| `ipywidgets`      | `>= 7.6.0`        | For interactive widgets in Jupyter notebooks               |
| `torchmetrics`    | `>= 0.10.0`       | For easy calculation of metrics during model training     |
| `optuna`          | `>= 2.10.0`       | For hyperparameter optimization                            |
| `fvcore`          | `>= 0.1.5`        | For utilities and helper functions                         |
| `iopath`          | `>= 0.1.9`        | For input/output path management                           |
| `submitit`        | `>= 1.3.0`        | For managing job submissions                               |
| `kornia`          | `>= 0.6.0`        | For computer vision tasks and transformations              |
| `prettytable`     | `>= 2.4.0`        | For pretty-printing tables                                 |
| `pickleshare`     | `>= 0.7.5`        | For a lightweight persistent storage                       |
| `torchcam`        | `>= 0.1.2`        | For visualization of neural network activations            |
| `torchattacks`    | `>= 0.2.0`        | For implementing adversarial attacks in PyTorch           |


#### DINOv2 Specific Requirements

for further information, check source [[2]](https://github.com/facebookresearch/dinov2)

| Library           | Version           | Why?                                                       |
|-------------------|-------------------|-----------------------------------------------------------|
| `torch`           | `== 2.0.0`        | Core deep learning library for building models             |
| `torchvision`     | `== 0.15.0`       | For vision-related utilities and transformations           |
| `omegaconf`       | `Latest`          | For managing configurations easily                         |
| `torchmetrics`    | `== 0.10.3`       | For easy calculation of metrics during model training     |
| `fvcore`          | `Latest`          | For utilities and helper functions                         |
| `iopath`          | `Latest`          | For input/output path management                           |
| `submitit`        | `Latest`          | For managing job submissions                               |
| `xformers`        | `== 0.0.18`       | For efficient transformers                                 |
| `cuml-cu11`       | `Latest`          | For GPU-accelerated machine learning                       |


## Data

Our project utilizes the [Best Alzheimer's MRI Dataset](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy), a comprehensive and curated collection of MRI scans specifically designed for Alzheimer's classification. This dataset serves as an improved version of the raw dataset available [here](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images).

To address class imbalance, the dataset incorporates synthetic images generated through advanced techniques, ensuring a more balanced distribution of classes. The original real images have been allocated to the test directory for thorough evaluation.

All images have been resized to **224x224 pixels** to meet the input requirements of the deep learning models used in this project. The data is organized as follows:


A **`prepare_dataset.ipynb`** notebook located in the **`dataset`** directory is responsible for splitting the data into:

- **80%** for training
- **10%** for validation
- **10%** for testing

The data is structured as follows:

- **Training Set**: Contains both synthetic and real images, providing a balanced source for model training.
- **Validation Set**: Used for hyperparameter tuning and monitoring model performance during training to help prevent overfitting.
- **Test Set**: Comprises only real images for an unbiased final evaluation of the trained models.

## Models

* We provide pre-trained checkpoints for the [dataset](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy) we used: the normal version and the under attack version. All model checkpoints should be placed inside the `/checkpoints` directory.
* For each model, the optuna parameters that were used for the training are also avaialble in a **json** file. In order to use the json file, place it under `/checkpoints/optuna_params`


| Model Type                 | Google Drive Link                                                                                | Optuna Params                                                                                |
|----------------------------|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `DINOv2 fine-tuned`          | [Download .pth file](https://drive.google.com/file/d/1jABpNVpMTrehBhL6ilm2e0gTx2C7fQyd/view?usp=drive_link)      | [Download JSON file](https://drive.google.com/file/d/1EyuLRh44TgUEpMn5ErN77XdZEmFZ8MNd/view?usp=drive_link) |
| `DINOv2_atk fine-tuned`                 | [Download .pth file](....)       | [Download JSON file](https://drive.google.com/file/d/1EyuLRh44TgUEpMn5ErN77XdZEmFZ8MNd/view?usp=drive_link) | 
| `Resnet34 fine-tuned`        | [Download .pth file](.....)       | [Download JSON file](....) | 
| `Resnet34_atk fine-tuned`               | [Download .pth file](....)       | [Download JSON file](....) | 
| `EfficientnetB0 fine-tuned`  | [Download .pth file](https://drive.google.com/file/d/1SuudU3uv2FCtJiD1XKdfpAPaMnk70Qhw/view?usp=drive_link)       | [Download JSON file](https://drive.google.com/file/d/1Bha09qvH3MlP112wm-j9j_NqijC5nPFL/view?usp=drive_link) | 
| `EfficientnetB0_atk fine-tuned`         | [Download .pth file](.....)        | [Download JSON file](https://drive.google.com/file/d/1Bha09qvH3MlP112wm-j9j_NqijC5nPFL/view?usp=drive_link) | 

## Results

## Usage

In this repository , you have the following usage options:

#### train a specific model

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

#### adversarial attacks: training and attacking
**Note:** This option can be performed only if you have a **pretrained model** with a defined architecture under the **`models/def_models.py`** file.

Here are the common steps for both options:
1. copy one of the models notebook **`dino_v2_model_atk.ipynb`** , **`efficienetnet_model_atk.ipynb`** ,**`resnet_model_atk.ipynb`**. rename the new notebook to be **`*_model_atk.ipynb`** where * is the name of your backbone.
2. set the models definition from the training process under no attacks.
3. set **`load_existing_hyperparams = True`** if you want to use the previous optuna hyper-parameters that you found in the first training process under no attacks. This is the recommended approach. Otherwise you can set **`load_existing_hyperparams = False`** and make a new optuna search.
4. Load the pretrained model. set the **`pretrained_model_path`** parameter to the path of your pretrained model.

##### perform attacks

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


##### adversarial attacks training

1. define the following parameters : `Epsilon`, `num_iter`, `Alpha`(see contraints under [PGD attack](#pgd-attack)), `adv_weight`, `num_epochs`.
2. define the number of epochs for the training.
3. define the augmentations if needed.
4. in the training process you have 2 options:
    *  Train from scratch. For this option , set the  **`load_existing_params = False`**. 
    * reload an exisiting model. you can reload an  existingmodel  that was previously trained and choose to   train itor not.  Loading an exisiting model   set**`load_existing_params =  True`**. for training a   loadedmodel, set the **`train_loaded_model = True`**  inaddition.
5. After the training process, the confusion matrix, loss curve and accuracy curve are saved under the `assets/*` when  `*` is the models name.


  

## Future Work

1. **Investigation advanced Targeted/Untargeted Attacks and Black-Box Attacks**: Future work can include investigating various targeted attacks, including advanced versions of FGSM such as Iterative FGSM (I-FGSM), Targeted I-FGSM, IND and OOD attacks, Kryptonite Attacks and one pixel attacks. These attacks can be performed in a targeted manner to evaluate the model's vulnerabilities and also in an untargeted way. Additionally, we can further explore black-box attacks when the model is not available to the attacker.

2. **Diverse Datasets**: In the future we plan to utilize a more diverse dataset featuring various MRI images across different demographics and orientations. Currently, we used the T1 MRI. we can incorporate the T2 and more...
We can also use a multi-modal data such as PET and CT scans.

3. **RNN Performance Evaluation**: We will assess the performance of Recurrent Neural Networks (RNNs) to determine their contribution to the robustness of the ensemble model.

4. **Ensemble Method Exploration**: Various ensemble techniques, such as stacking and blending.

5. **Real-World Testing**: Conducting real-world tests in collaboration with radiologists and people that are responsible for the reliability of medical data - images in particular. This can check the model's applicability and his reliability in the real world.




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
