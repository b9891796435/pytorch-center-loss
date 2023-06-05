import cv2
import imutils
import numpy as np
from torchvision.transforms import ToTensor
import torch
from datasets import format_img
from models import A_mobileNet
from collections import OrderedDict
from PIL import Image

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
# emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
emotion_model_path = 'model.pth'

# hyperparameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_dict = torch.load(emotion_model_path, map_location=torch.device('cpu')).eval().state_dict()
emotion_classifier = emotion_dict.items()
temp_arr = []
for i in emotion_classifier:
    temp_item = list(i)
    temp_item[0] = temp_item[0][14:]
    temp_arr.append(temp_item)
emotion_classifier = A_mobileNet(8)
emotion_dict = OrderedDict(temp_arr)
emotion_classifier.load_state_dict(emotion_dict)
emotion_classifier.eval()
# EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
#             "neutral"]
EMOTIONS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust",
            "fear", "contempt"]

# feelings_faces = []
# for index, emotion in enumerate(EMOTIONS):
# feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

# starting video streaming

def frame_parse(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
                       key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
        # the ROI for classification via the CNN
        res = []
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = cv2.resize(roi.astype('uint8'), (224, 224), interpolation=cv2.INTER_CUBIC)
        res.append(roi)
        res = np.array(res)
        res = np.expand_dims(res, -1)
        res = np.repeat(res, 3, axis=-1)
        res = res.squeeze(0)
        res = Image.fromarray(np.uint8(res), mode='RGB')
        res = ToTensor()(res)
        res = res.unsqueeze(0)

        with torch.no_grad():
            preds = emotion_classifier(res)[0]
            preds = preds[0].numpy()
            preds = preds.T
            preds = np.exp(preds) / np.sum(np.exp(preds), axis=0)
            preds = preds.T
        return preds, faces
    else:
        return None
if __name__ == '__main__':
    cv2.namedWindow('your_face')
    camera = cv2.VideoCapture(2)
    while True:
        frame = camera.read()[1]
        frame = imutils.resize(frame, width=300)
        frameClone = frame.copy()
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        # reading the frame
        result=frame_parse(frame)
        if result is not None:
            (preds, faces) = result
            (fX, fY, fW, fH) = faces
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
        else:
            continue

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            # draw the label + probability bar on the canvas
            # emoji_face = feelings_faces[np.argmax(preds)]

            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)
            cv2.putText(frameClone, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                          (0, 0, 255), 2)
        #    for c in range(0, 3):
        #        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
        #        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
        #        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)

        cv2.imshow('your_face', frameClone)
        cv2.imshow("Probabilities", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
