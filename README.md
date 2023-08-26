# LAS
This repository includes an implementation of the Listen Attend and Spell model that transcribes speech utterences to characters. The model is composed of a  listener, an attention module and a speller. The listener is a pyramidal recurrent network encoder(PBLSTM) that takes filter bank spectra as inputs. The speller is an attention-based recurrent network decoder that outputs characters. The model uses greeedy decoding. The full paper can be found here [Listen Attend and Spell](https://arxiv.org/abs/1508.01211). 
### Contents
#### src
* **main.py** Use it to run training, validation, and inference
* **model.py** File contains the LAS model. It contains the Listener, Attention and Speller Module.
* **train_test.py** File contains the train, validation and test code.
* **dataloader.py** File contains code to load data for training the model.
#### config
* **config.yaml** File contains model and dataset configurations 
#### Notebook
* **notebook.ipynb** Notebook for running the model. The data should be placed in a data folder with a split for training, validation and testing

### Usage
Download and unzip the dataset into a data folder. Use python main.py to run the model. Use the config.yaml file to change model parameters

