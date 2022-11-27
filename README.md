# ad-block-real-world

This project aims to create an object detection model capable of automatically detecting advertsiment on images.

Dataset available in:
https://mekabytes.com/dataset/info/billboards-signs-and-branding

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── annotations    <- Annotations.
    │   └── images         <- The original images.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── dataset        <- Datasets classes
        │   └── ads_dataset.py
        │
        ├── predict_model.py
        └── train_model.py

---

References:

https://www.kaggle.com/ipythonx/keras-global-wheat-detection-with-mask-rcnn/notebook
