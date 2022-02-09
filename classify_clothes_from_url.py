from PIL import Image
from urllib import request
from io import BytesIO
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--url", required=True)
    return parser.parse_args()


def get_image_from_url(url):
    res = request.urlopen(url)
    img = Image.open(BytesIO(res.read())).resize((180, 180))
    return img


def classify_image(img):
    model = keras.models.load_model("clothes_classifier_vgg16_with_augmentation_and_fine_tuning.keras")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.vgg16.preprocess_input(x)

    preds = model.predict(x)
    with open("labels.json", "r") as file:
        labels = json.load(file)
    print(f"{labels[str(np.argmax(preds))]}: {np.max(preds):.2%}")


def main():
    args = parse_args()
    img = get_image_from_url(args.url)
    classify_image(img)


if __name__ == "__main__":
    main()
