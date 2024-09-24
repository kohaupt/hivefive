# Project: Predicting the health of bee colonies based on audio

This repository provides the code for the project as part of the ‘Audio Data Science’ module at Düsseldorf University of Applied Sciences. The aim of the study is to determine the health of bee colonies based on the presence of a queen bee using audio recordings and machine learning. The analyses and processing carried out here are based on the work of Nolasco et al. [\[1\]](#Nolasco19). The corresponding data set is available online on [zenodo](https://zenodo.org/records/2667806) (accessed on 2024-09-24).

## Project Structure

The folder structure is loosely based on [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/). The folders should be used as follows:

```
├── data*
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
|
├── notebooks          <- Jupyter notebooks
│
├── models**           <- Trained and serialized models, model predictions, or model summaries
│
└── src                <- Source code for use in this project.
    │
    └── modeling       <- Scripts to train models and then use trained models to make predictions
```

\*Note: The data folder is not included in this repository.  
\*\*Note: The trained models are accessible via [Zenodo](https://zenodo.org/).

## Usage

The steps of exploration, training of the ML-models and evaluation can be carried out using the associated jupyter notebooks. The jupyter notebooks access classes in the `./src` folder for this purpose. The following (fundamental) technologies and libraries are used for this and are necessary for execution (non-exhaustive list):

### Programming languages/environments:

- Python
- Anaconda
- Jupyter Notebook

### Libraries:

- librosa
- scikit-learn
- pandas
- pytorch

Before running the jupyter notebooks, make sure that the paths in the `./src/config.py` file correspond to your local folder structure.

## Pipelines

The content and functionality of the three jupyter notebooks are briefly described below. Note: The notebook `modelling.ipynb` is located in the `.src/modeling` folder due to the necessary imports of classes in surrounding folders, contrary to the above-mentioned convention.

### ./notebooks/exploration.ipynb

This Jupyter notebook performs a basic data exploration on the dataset. To do this, the notebook accesses the class `./src/features.py` and performs preprocessing and segmentation on the audio files located in `./data/raw`.

### ./src/modeling/modeling.ipynb

In this Jupyter notebook, four ML-models are trained and the corresponding ML-models are exported as `.pt` files (pytorch) in `./models/final`. In each epoch, a new model is persisted if, according to the loss function, it performs better than any previous model. For each persisted model, the model itself is saved in the schema `model_<<date>>_<<time>>_<<epoch>>.pt`. At the same time, a file with the additional suffix `_checkpoint.pt` is persisted, which stores the parameters of the selected optimizer and the number of epochs in addition to the actual model. This allows trained models to be read in, retrained and analysed at a later date.

### ./notebooks/evaluation.ipynb

In this Jupyter notebook, the ML-models calculated as part of the study are evaluated using metrics such as confusion matrices and ROC-curves.

## References

<a name="Nolasco19"></a> [1] I. Nolasco, A. Terenzi, S. Cecchi, S. Orcioni, H. L. Bear, and E. Bene-tos, “Audio-based identification of beehive states,” in ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing(ICASSP). IEEE, 2019, pp. 8256–8260.
