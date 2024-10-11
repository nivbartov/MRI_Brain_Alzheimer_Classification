<h1 align="center">MRI Brain Alzheimer Classification Under Adversarial Attacks</h1>

<div style="text-align: center;">
<h1>
  <img src="assets/icons/main_image_readme.jpg" height="300">
</h1>

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
- [Model](#model)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [Sources and References](#sources-and-references)
- [Citation](#citation)
- [License](#license)


## Project Overview

## Project Structure
```
MRI_Brain_Alzheimer_Classification/
├── 🖼️ assets/
│   ├── 🔍 Dinov2/
│   │   ├── confusion_matrix.png
│   │   ├── loss_curve.png
│   │   └── accuracy_curve.png
│   ├── 🔍 Dinov2_atk/
│   │   ├── ... 
│   ├── 🔍 Efficientnet/
│   │   ├── ...
│   ├── 🔍 Efficientnet_atk/
│   │   ├── ...
│   ├── 🔍 Restnet/
│   │   ├── ...
│   └── 🔍 Resnet_atk/
│       ├── ...
├── ⛓️ checkpoints/
│   ├── 🎯 optuna/
│   │   ├── Dinov2/
│   │   ├── Dinov2_atk/
│   │   ├── Efficientnet/
│   │   ├── Efficientnet_atk/
│   │   ├── Restnet/
│   │   └── Resnet_atk/
│   ├── Dinov2/
│   │   ├── ...
│   ├── Dinov2_atk/
│   │   ├── ...
│   ├── Efficientnet/
│   │   ├── ...
│   ├── Efficientnet_atk/
│   │   ├── ...
│   ├── Restnet/
│   │   ├── ...
│   └── Resnet_atk/
│       ├── ...
├── 📊 dataset/
│   ├── dataset_variables/
│   │   ├── train_set.pt
│   │   ├── validation_set.pt
│   │   └── test_set.pt
│   ├── raw_dataset/
│   │   ├── train/
│   │   └── test/
│   └── prepare_dataset.ipynb
├── 🌐 env/
│   └── project_env.yaml
├── 📚 models/
│   ├── results.ipynb
│   ├── Dinov2.ipynb
│   ├── Dinov2_atk.ipynb
│   ├── Efficientnet.ipynb
│   ├── Efficientnet_atk.ipynb
│   ├── Restnet.ipynb
│   └── Resnet_atk.ipynb
└── 🛠️ utils/
    ├── optuna_search.py
    ├── utils_funcs.py
    └── grad_cam.py
```

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
| `utils/gradcam.py` | Script for generating Grad-CAM heatmaps. Requires a model with a convolutional layer. Users must specify parameters as per function definitions. |
| `utils/utils_funcs.py` | Contains general utility functions for tasks such as saving models, loading images, displaying graphs, and training. Includes specific functions for adversarial training. |
| `models/*_model.ipynb` | Model-specific notebook (e.g., `resnet_model.ipynb`). Covers the complete process: data loading, training, saving, and evaluation. Generates confusion matrices and plots loss/accuracy curves. |
| `models/*_model_atk.ipynb` | Adversarial training notebook for each model (e.g., `resnet_model_atk.ipynb`). Loads a pretrained model, applies adversarial attacks, and trains the model on these attacks. |


## Installation

This code works with any OS — Linux or Windows. To set up all the required dependencies, please follow one of the instructions below:

#### Conda

[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) (Recommended) - Clone this repository and then create and activate the **`env/project_env.yaml`** conda environment using the provided environment definition:

```
conda env create -f env/project_env.yaml
conda activate project_env
```

#### Pip Install

Clone this repository and then use the provided **`env/requirements.txt`** to install the dependencies.

```
pip install -r env/requirements.txt
```

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

## Model

## Results

## Usage

## Future Work

## Future Work

1. **Investigation advanced Targeted/Untargeted Attacks and Black-Box Attacks**: Future work can include investigating various targeted attacks, including advanced versions of FGSM such as Iterative FGSM (I-FGSM), Targeted I-FGSM, IND and OOD attacks, Kryptonite Attacks and one pixel attacks. These attacks can be performed in a targeted manner to evaluate the model's vulnerabilities and also in an untargeted way. Additionally, we can further explore black-box attacks when the model is not available to the attacker.

2. **Diverse Datasets**: In the future we plan to utilize a more diverse dataset featuring various MRI images across different demographics and orientations. Currently, we used the T1 MRI. we can incorporate the T2 and more...
We can also use a multi-modal data such as PET and CT scans.

3. **RNN Performance Evaluation**: We will assess the performance of Recurrent Neural Networks (RNNs) to determine their contribution to the robustness of the ensemble model.

4. **Ensemble Method Exploration**: Various ensemble techniques, such as stacking and blending.

5. **Real-World Testing**: Conducting real-world tests in collaboration with radiologists and people that are responsible for the reliability of medical data - images in particular. This can check the model's applicability and his reliability in the real world.




## Sources and References

#### Sources

1. Facebook Research. (2023). DINOv2. GitHub repository. [https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

2. Luke Chugh. (2021). Best Alzheimer MRI dataset. Kaggle dataset. [https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy)

3. Gil, J. (2020). PyTorch Grad-CAM. GitHub repository. [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

4. Optuna. (2023). Optuna: A hyperparameter optimization framework. GitHub repository. [https://github.com/optuna/optuna](https://github.com/optuna/optuna)

#### References

1. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Understanding adversarial attacks on deep learning based medical image analysis systems. [arXiv:1907.10456](https://arxiv.org/abs/1907.10456)

2. Zhang, Y., & Yu, L. (2023). Adversarial attacks on foundational vision models. [arXiv:2308.14597](https://arxiv.org/abs/2308.14597)

3. Chen, X., Zhang, H., & Li, Y. (2022). Exploring adversarial attacks and defenses in vision transformers trained with DINO. [arXiv:2206.06761](https://arxiv.org/abs/2206.06761)

4. Xie, L., & Wang, Z. (2023). DINOv2: Learning robust visual features without supervision. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)


5. Tajbakhsh, N., Shin, J., Gurudu, S. R., & Hurst, R. T. (2022). What makes transfer learning work for medical images: Feature reuse & other factors. [arXiv:2203.01825](https://arxiv.org/abs/2203.01825)

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
