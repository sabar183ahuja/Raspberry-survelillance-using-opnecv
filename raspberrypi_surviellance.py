import numpy as np
import cv2, time
import threading
import pickle
from datetime import datetime
import sender


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640,480))

# load haar cascades for frontal face identifer
lastUploaded = datetime.now()
first_frame=None
motionCounter = 0
counterLimit  = 100
minContourArea = 10000
images = []
frames = []
master ="sabar"
probability = 0.8
lock= threading.Lock()
print("[INFO] loading face detector...")

# load our serialized face detector from disk
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# # alternate for above caffe model
# modelFile = "opencv_face_detector_uint8.pb"
# configFile = "opencv_face_detector.pbtxt"
# these models can be found online
# net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open( "output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())
# time to allow camera to warn-up
time.sleep(2)

"""
target function for thread
Thread is required to avoid the surevillance from stopping while the frames are being processed for motions and faces
"""
def WriteANDSendVideo(frames,lock):
    lock.acquire()
    send = True
    for frame in frames:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < probability:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
            	continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
            	(96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            # if master enters room no need to send alert
            if( name == master):
                send = False
                break
            # draw the bounding box of the face along with the
            # associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
            	(0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
            	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    if(send):
        height,width,layers=frames[1].shape
        out=cv2.VideoWriter("video.mp4",-1, camera.framerate,(width,height))
        for frame in frames: # We iterate on the frames of the output video:
            out.write(frame) # We add the next frame in the output video.
        sender.send_email()
        lastUploaded = timestamp
        frames.clear()
        out.release()


for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = f.array
    timestamp = datetime.now()
    status=0
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame= gray
        continue

    delta_frame=cv2.absdiff(first_frame,gray)
    thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

    (cnts,b)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        # if contour is too small skip
        if cv2.contourArea(contour) < minContourArea:
            # this signifies that any existing motion has now ceased
            if motionCounter>= counterLimit:
                motionCounter = 0
                thread= threading.Thread(target=WriteANDSendVideo ,args=(images.copy(),lock))
                thread.start()
                images.clear()
            continue


        # check to see if enough time has passed between uploads
        if (timestamp - lastUploaded).seconds >= 1 :
            cv2.putText(frame, "TimeStamp"+str(timestamp), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.imwrite("talkingraspi_{}.jpg".format(motionCounter), frame);
            motionCounter+=1
            images.append(frame)

        else:
            motionCounter =0
            images.clear()

    # cv2.imshow("Frame",frame)

    key=cv2.waitKey(1)
    rawCapture.truncate(0)
    if key==ord('q'):
        break

# cv2.destroyAllWindows
