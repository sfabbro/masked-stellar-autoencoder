## batch_scripts

Some example scripts for training on a HPC clusters such as with [Narval](https://www.alliancecan.ca/en/services/compute/narval) provided by the
Digital Research Alliance of Canada. 

---

### Usage

Given your credentials and environment are used in editing these files, ```msa_init.slurm``` is run first to 1. Give an idea of how long to set
the training time of each epoch when looping and 2. Create the initial save file for the model. If epoch pre-training time is known and training 
from a presaved model (inserted into ```presaved``` in ```configs/pretrain.yaml```), ```msa_looping.slurm``` can be run to pre-train the model over many epochs. 

<!-- This manual initialization can be improved by adding a Try-Except when loading the saved weights in ```model.py```, removing the need of the
msa_init.slurm, to be added in the future after handling more pressing tasks as training is not impeded. -->
