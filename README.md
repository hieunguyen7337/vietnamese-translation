# Speech recognition vietnamese
## Description 
This is a Python repo for vietnamese-english translation using the `envit5-translation` model develop by VietAI.

## Requirements 
- Python 3.9

## Installation 
Run the following command to install library 

`pip install -r requirements.txt`

to download the envit5 model file run the following command if the terminal is linux

`wget -P ./envit5_model https://huggingface.co/VietAI/envit5-translation/resolve/main/pytorch_model.bin`

if the terminal is window, use the following command

`Invoke-WebRequest -Uri https://huggingface.co/VietAI/envit5-translation/resolve/main/pytorch_model.bin -OutFile ./envit5_model/pytorch_model.bin`

## Usage
You can run the inference of the `envit5-translation` model with the `infer.py` file.

## Credits
This repo uses the envit5-translation model developed by VietAI, coupled with the Hugging Face Transformers library, which is an open-source library for NLP models.