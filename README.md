# Clothes classifier (using Keras)
## About
- This small project uses keras built in VGG16 classifier model to classify clothing type given url with the picture. 
- It works only for clasess listed in labels.json.
- Only one type of clothing needs to be present in the picture.
- Human can't be present in the picture.

## Prerequisites
```bash
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## About the model
- VGG16 model with pretrained weigths on imagenet dataset
- https://www.kaggle.com/agrigorev/clothing-dataset-full dataset used
- Data augmentation and fine tuning techniques have been used
- Current validation accuracy is around 80% due to small and unbalanced dataset.

## How to train the model
- Setup kaggle api: https://www.kaggle.com/docs/api (only needed for training)
- Run Jupyter notebook train_clothes_classifier.ipynb

## How to test the model
```bash
python classify_clothes_from_url.py --url=https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/f694561a-1c8e-42f8-8997-79251ccdc468/bucket-hat-Jj8GT4.png
```
Expected output:
```bash
Hat: 100.00%
```
