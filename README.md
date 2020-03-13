# emnist-pytorch
Handwritten character recognition using Pytorch and EMNIST

## Environment and prerequisites

Create a conda environment with the packages listed in environment.yml.

To create an environment with the configuration from environment.yml, use the following command:

```
conda env create -f environment.yml
```

A new conda environment called "emnist" will be created.

Activate the conda environment:

```
conda activate emnist
```

## Data preparation

The emnist handwritten character recornition dataset will be automatically 
downloaded by the code and the data will be organized into appropriate folder for pytorch to access.

It may take some minutes (~30) to organize the data. 
As an alternative, you can download the organized data zip file from https://drive.google.com/drive/folders/1JLE0kz9ctZ4HI2vA6gZbec1MY1so5QLK?usp=sharing
and extract it in the folder `./data/emnist`.

The data directory should have the following structure:

```
./data/emnist
   - train
      - 0/<images>
      - 1/<images>
      ...

   - test
      - 0/<images>
      - 1/<images>
      ...

```


## Training and Testing

To train and test, run the following:
```
cd src/
python doall.py
```

The checkpoints are saved under the folder `./scratch/`.


## Credits

Imbalanced Data Sampler is copied from https://github.com/ufoym/imbalanced-dataset-sampler
