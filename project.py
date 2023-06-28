import cv2
from tensorflow import keras
import tensorflow
from tensorflow import keras
from keras.models import load_model
import sys
import numpy as np
import time

# converting image to grayscale


def convertToGrayScale(img):
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g_img = cv2.GaussianBlur(g_img, (5, 5), 0)
    g_img = cv2.adaptiveThreshold(
        g_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=321, C=28)
    return g_img


def predict(contours, Mnist_model, gray_img, img):
    print(len(contours))
    for i in range(len(contours)):
        [x, y, w, h] = cv2.boundingRect(contours[i])
        print(x, y, w, h)
        if (w > 3) & (h > 10):
            pad = 15
            print("yaxis,xaxis", y-pad, y+h+pad, x-pad, x+w+pad)
            crop_image = gray_img[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
            cv2.imwrite(f'output_images/img{i}.jpg', crop_image)
            crop_image = crop_image/255.0

            crop_image = cv2.resize(crop_image, (28, 28))
            pred = Mnist_model.predict(
                crop_image.reshape(1, 28, 28, 1)).argmax()
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            print(pred)
            prediction = str(pred)
            cv2.putText(img, prediction, (x, y-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return img


def processImage(ret, frame, Mnist_model, out):
    if ret == True:
      img = cv2.resize(frame, (256, 256))
      g_img = convertToGrayScale(img)
      thresh = cv2.Canny(g_img, 50, 100)
      contours, hierarchy = cv2.findContours(
          thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      predict(contours, Mnist_model, g_img, img)
      out.write(img)
      cv2.imshow("Vaishak : Output", img)


def main():
    print("press 1 for input video , 2 for live video")
    user_option = input()
    start_time = time.time()
    Mnist_model = load_model('model_tarin')

    if (user_option == "1"):
        cap = cv2.VideoCapture('Govind_Chennu_Testing.mp4')
        # Check if camera opened successfully
        if (cap.isOpened() == False):
          print("unable to open video file!")

        target_size = (256, 256)
        # Define codec and create VideoWriter object
        out = cv2.VideoWriter('processed_video_output.avi', cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 5, target_size)

        while (cap.isOpened() and time.time()-start_time < 20):
            ret, frame = cap.read()
            processImage(ret, frame, Mnist_model, out)

            # j+=1
            if cv2.waitKey(20) & 0xFF == ord('q'):
              break

    else:
        print("live_video_Recording")
        cap = cv2.VideoCapture(0)
        if (cap.isOpened() == False):
            print("Unable to read camera!!!")

        # Mnist_model = load_model('model_tarin.h5')
        out = cv2.VideoWriter('webcam_live_output.avi', cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 5, (256, 256))

        while (cap.isOpened() and int(time.time()-start_time) < 20):
            ret, frame = cap.read()
            processImage(ret, frame, Mnist_model, out)

            if cv2.waitKey(20) & 0xFF == ord('q'):
              break


if __name__ == "__main__":
    main()
