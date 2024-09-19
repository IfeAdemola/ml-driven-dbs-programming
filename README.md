# Machine Learning-Driven Deep Brain Stimulation Programming

This repository is the official implementation of An Investigative Study Exploring Machine Learning Approaches for Optimising Deep Brain Stimulation Programming

## Requirements
To install reqirements:

```setup
pip install -r requirements.txt
```
## Training
To train the model(s) in the paper, run this command:

```train
python train.py --data_path <path_to_data> --space <mni> --target 
```
## Evaluation
To evaluate my model on ImageNet, run:

```eval
python inference.py --model-path mymodel.pth --ct_path <path_to_processor_transformer_file> --data_path <path_to_data> --space <mni>
```
