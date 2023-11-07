# MADANet: A Lightweight Hyperspectral Image Classification Network

## Overview
This is a project about **MADANet**, a lightweight hyperspectral image classification network that combines multi-scale feature aggregation and dual attention mechanism. The goal of this project is to implement this network and test it on various hyperspectral image datasets.

## Article Citation
Cui, B.; Wen, J.; Song, X.; He, J. MADANet: A Lightweight Hyperspectral Image Classification Network with Multi-Scale Feature Aggregation and Dual Attention Mechanism. Remote Sens. 2023, 15, 5222. https://doi.org/10.3390/rs15215222

## Project Structure
- `.idea/`: Folder for IDE settings
- `.gitignore`: Lists files and folders to be ignored by git
- `CBAM.py`: Contains code for Convolutional Block Attention Module (CBAM)
- `DANet.py`: Contains code for Dual Attention Network (DANet)
- `Indian_pines_corrected.mat`: File for the Indian Pines dataset
- `Indian_pines_gt.mat`: File for the ground truth labels of the Indian Pines dataset
- `LICENSE`: Project license file
- `MADANet.py`: Contains code for the MADANet network
- `dataset`: Contains code for handling the dataset
- `evaluate.py`: Contains code for evaluating the model performance
- `train.py`: Contains code for training the model

## How to Run
1. Clone this repository
2. Download and prepare your datasets
3. Run `train.py`

## Results
Our model has only 0.16 M model parameters on the Indian Pines (IP) dataset, but the overall accuracy is as high as 98.34%. Similarly, the framework achieved overall accuracies of 99.13%, 99.17%, and 99.08% on the University of Pavia (PU), Salinas (SA), and WHU Hi LongKou (LongKou) datasets, respectively.

## Contributions
If you have any suggestions or issues for this project, feel free to raise an issue or pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
