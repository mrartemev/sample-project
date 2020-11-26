# Hair-recolor GAN, PyTorch

Hair recoloring GAN with hair segmentation built in.

#### Usage

1. Build conda environment:
    `conda env create -f environment.yml`
2. Run:
    `python main.py [args]`
3. Check comet.ml logs:
    `https://www.comet.ml/mrartemev/stgans/your_experiment_url` 

#### Config

Config in mainteined by [hydra](hydra.cc). 
Please read the quick guide for clarity

#### Todo:

1. Make discriminator fully convolutional (for some reason it wasn't working when I tried it)
2. Balance loss weights
