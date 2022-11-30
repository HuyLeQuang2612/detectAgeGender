import numpy as np
import tensorflow as tf
from PIL import ImageDraw


def getGender(pred_sex):
    index_maxValue = pred_sex.argmax()
    if index_maxValue == 0:
        return "Male"
    return "Female"


def getRangeAge(pred_age):
    arrRangeAge = ["0-14", "14-25", "25-40", "40-60", "60-116"]
    maxIndex = list(pred_age[0]).index(max(pred_age[0]))
    rangeAge = arrRangeAge[maxIndex] + " years old"
    return rangeAge


def draw_box_faces(image, faces):
    draw = ImageDraw.Draw(image)
    for face in faces:
        left = face["box"][0]  # left coordinate
        top = face["box"][1]  # top coordinate
        width = face["box"][2]  # face width
        height = face["box"][3]  # face height

        points = (
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height),
            (left, top)
        )
        draw.line(points, fill='#00d400', width=2)


def crop_faces(image, faces):
    images_crop = []
    for face in faces:
        x1 = face["box"][0]
        y1 = face["box"][1]
        x2 = (face["box"][0] + face["box"][2])
        y2 = (face["box"][1] + face["box"][3])
        image_crop = image.crop((x1, y1, x2, y2))
        images_crop.append(image_crop)

    return images_crop


def predict(arrImg):
    arr3d = np.zeros((48, 48, 3))
    arr3d[:, :, 0] = arr3d[:, :, 1] = arr3d[:, :, 2] = arrImg
    pixels = arr3d

    model = tf.keras.models.load_model('saved_model/efficientNetB2_weight.h5')
    pred=model.predict(np.array([pixels]))
    pred_age = pred[0][0]
    pred_gender = pred[1][0]

    print("\nPredicting Age:")
    print(f"0-14: {round(pred_age[0] * 100, 2)}%, 14-25: {round(pred_age[1] * 100, 2)}%, 25-40: {round(pred_age[2] * 100, 2)}%, 40-60: {round(pred_age[3] * 100, 2)}%, 60-116: {round(pred_age[4] * 100, 2)}%")
    print("Predicting Gender:")
    print(f"Male: {round(pred_gender[0] * 100, 2)}%, Female: {round(pred_gender[1] * 100, 2)}%\n")

    age = getRangeAge(pred[0])
    gender = getGender(pred[1])

    return age, gender
