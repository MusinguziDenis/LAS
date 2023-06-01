# LAS
This repo provides an implementation of the Listen Attend and Spell model that transcribes speech utterences to characters. The model is composed of a  listener and a speller. The listener is a pyramidal recurrent network encoder that accepts filter bank spectra as inputs. The speller is an attention-based recurrent network decoder that emits characters as outputs. The model uses greeedy decoding. The full paper can be found here [Listen Attend and Spell](https://arxiv.org/abs/1508.01211). The following listing should give you an overview about the files/directories.
* **main.py** Use it to run training, validation, and inference
* **model.py** File contains the LAS model. It contains the Listener, Attention and Speller Module.
* **train_test.py** File contains the train, validation and test code.
* **dataloader.py** File contains code to load data for training the model.
* **notebook.ipynb** Notebook for running the model. The data should be placed in a data folder with a split for training, validation and testing
