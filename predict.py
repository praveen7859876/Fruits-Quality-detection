import keras
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

print("MODULES IMPORTED")

# Config
custom_model_path = "models/inception_v3/inception_model_10_epochs.h5";
custom_testing_images_path = "testing_images";
classes = ['fresh apples', 'fresh banana', 'fresh oranges', 'rotten apple', 'rotten banana', 'rotten orange']
confidence_threshold = 0.90

model = keras.models.load_model(custom_model_path)

print("MODEL SUMMARY\n")
# print(model.summary())
print("Layers in model: ", len(model.layers))

def getColor(predicted_class):
    fresh_color = (0, 255, 0)
    rotten_color = (0, 0, 255)
    if(predicted_class in classes[:3]):
        return fresh_color
    else:
        return rotten_color

def get_pre_processed_img(image_test):
    img = Image.fromarray(image_test, 'RGB')
    img = img.resize((150, 150))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    return img

def start_video_capture():
    vid = cv2.VideoCapture(0)

    while(True):
        ret, frame = vid.read()

        img = get_pre_processed_img(frame)
        prediction = model.predict(img)
        confidence_score = np.amax(prediction)
        if (confidence_score > confidence_threshold):
            predicted_class = classes[np.argmax(prediction)]
        else:
            predicted_class = "No Fruit Detected"

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2
        color = (255, 0, 0)

        image = cv2.putText(frame , f"{predicted_class.upper()}" , (50, 50), font, fontScale,
            getColor(predicted_class),
            thickness, cv2.LINE_AA)
        image = cv2.putText(frame , f"{int(confidence_score * 100)} %", (50, 100), font, fontScale,
            getColor(predicted_class),
            thickness, cv2.LINE_AA)
        cv2.imshow('Fruit Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

def predict_custom_images():
    print("Predicting...")

    file_names = []
    predictions = []

    for file_name in os.listdir(custom_testing_images_path):
        im_path =os.path.join(custom_testing_images_path, file_name)
        image_test = cv2.imread(im_path)
        img = get_pre_processed_img(image_test)

        prediction = model.predict(img)
        confidence_score = np.amax(prediction)
        if (confidence_score > 0.90):
            predicted_class = classes[np.argmax(prediction)]
        else:
            predicted_class = "No Fruit Detected"

        file_names.append(file_name)
        predictions.append(predicted_class)

    data = {"File Name": file_names, "Prediction": predictions}
    df = pd.DataFrame(data)

    print(df.to_string(index=False))


def main():
    predict_custom_images()
    start_video_capture()

if __name__ == '__main__':
    main()
