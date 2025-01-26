# Diabetic Retinopathy Detection

This project aims to develop a machine learning model capable of detecting diabetic retinopathy from retinal images.

## Table of Contents

- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Dataset Separation](#2-dataset-separation)
  - [3. Model Training](#3-model-training)
- [Utility Functions](#utility-functions)
- [Contributors](#contributors)

---

## Project Structure

The repository contains the following files and directories:

- `label-rule.json`: Contains labeling rules for the dataset.
- `preprocess.py`: Script for preprocessing images.
- `seperate-dataset.py`: Script to separate the dataset into training and validation sets.
- `train.py`: Script to train the machine learning model.
- `utility`: Directory containing utility functions and modules:
  - `model.py`: Defines the machine learning model architecture and training functions.

---

## Setup Instructions

To set up the project on your local machine, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/abiwinanda/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

### 3. Install Dependencies

Ensure you have the required Python packages installed. Since the repository doesn't provide a `requirements.txt`, you may need to install the dependencies manually based on the imports in the scripts.

For example:

```bash
pip install numpy pandas tensorflow opencv-python
```

---

## Usage

The project involves three main steps: data preprocessing, dataset separation, and model training.

### 1. Data Preprocessing

Run the `preprocess.py` script to preprocess the retinal images. This may include resizing, normalization, and other image enhancement techniques.

```bash
python preprocess.py
```

Ensure that the script points to the correct directory containing your raw images.

### 2. Dataset Separation

Separate the dataset into training and validation sets using the `seperate-dataset.py` script.

```bash
python seperate-dataset.py
```

This script organizes your data into appropriate directories for training and validation.

### 3. Model Training

Train the machine learning model using the `train.py` script.

```bash
python train.py
```

The training script utilizes the model architecture defined in `utility/model.py`.

---

## Utility Functions

The `utility/` directory contains helper functions and modules. Below is an overview of the most important module:

- **`model.py`**
  - Defines the model architecture.
  - Includes training-related functions such as:
    - `train_model()`: Handles the training loop, including forward propagation, loss calculation, and backpropagation.
    - `get_trainable_params()`: Identifies and returns the parameters of the model that should be updated during training.

Refer to [`model.py`](https://github.com/abiwinanda/diabetic-retinopathy-detection/blob/master/utility/model.py) for details.
