# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
from playsound import playsound

from tkinter import *
import winsound

top = Tk()
top.geometry("760x700")
top.resizable(False, False)
top.title("MASK_DETECTION")

bg = PhotoImage(file='final.png')

label1 = Label(top, image=bg)

label1.place(x=0, y=0, relwidth=1, relheight=1)


def playThread():
    global t1
    t1 = threading.Thread(target=play())


def play():
    playsound('click.mp3')


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


def Recognition():
    # load our serialized face detector model from disk
    prototxtPath = r"face_detector\deploy.prototxt"
    weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = load_model("mask_detector.model")

    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()

    # loop over the frames from the video stream
    img_counter = 0

    while True:

        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 600 pixels
        frame = vs.read()

        frame = imutils.resize(frame, width=600)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            # label = "Mask" if mask > withoutMask else "without mask"

            # color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            # label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # playsound('danger.mp3') if withoutMask > mask else 'Mask'
            jj=False
            if mask > withoutMask:
                label = 'Mask'
                color = (0, 255, 0)
                label = "{}: {:.2f}%".format(
                    label, max(mask, withoutMask) * 100)
            
            elif withoutMask > mask:
                label = 'without mask'
                jj=True
                color = (0, 0, 255)
                label = "{}: {:.2f}%".format(
                    label, max(mask, withoutMask) * 100)
                    
                # if jj:
                #     winsound.Beep(1000, 1000)
                #     break
                    

                # img_name = "images\opencv_frame_{}.png".format(img_counter)
                # cv2.imwrite(img_name, frame)

                # img_counter += 1

            else:
                print('problem detected')

                break

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame

        cv2.imshow("MASK_DETECTOR", frame)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        if cv2.getWindowProperty('MASK_DETECTOR', cv2.WND_PROP_AUTOSIZE) < 1:
            break

    # do a bit of cleanup
cv2.destroyAllWindows()


def kill():
    global start
    start = False
    play()
    top.destroy()


function = False


act_b = PhotoImage(file='bb2.png')
label1 = Label(image=act_b)
Activating_button = Button(top, image=act_b, command=lambda: [play(), Recognition()], borderwidth=0,
                           width=130,
                           height=130,
                           bg="black", activebackground="black").place(x=50, y=560)

stop_button = PhotoImage(file='stop.png')
label2 = Label(image=stop_button)

stop_b = Button(top, image=stop_button, command=lambda: [kill()], borderwidth=0,
                bg='black', activebackground='black').place(x=630, y=585)

top.mainloop()
