
import datetime
import numpy as np
import pygame
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import os



# mengambil model detektor wajah dari disk
prototxtPath = os.path.sep.join(["./face_detector/", "deploy.prototxt"])
weightsPath = os.path.sep.join(["./face_detector/", "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# mengambil model detektor masker wajah dari disk
maskNet = load_model("C:/Users/User/Belajar_ML/final_mask_detection/web_demo/static/models/mask_model.h5")


# Tambahan musik 
pygame.mixer.pre_init(44100, -16, 2, 2048) # atur mixer untuk menghindari jeda suara
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load('C:/Users/User/Belajar_ML/final_mask_detection/sounds/beep.mp3')



class MaskDetection:

    def __init__(self) :
        # capturing camera
        self.video = cv2.VideoCapture(0)


    def __del__(self) :
        self.video.release()
        
    def detect_and_predict_mask(frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (800, 800),
            (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

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
                if face.any():
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (96, 96))
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


    def get_mask(self):
        #extraksi frame
        ret, frame = self.video.read()
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)

        locs, preds = MaskDetection.detect_and_predict_mask(frame, faceNet, maskNet)

        #Time
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        counter_masks = {'has_mask': 0,'no_mask': 0}

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            if (label == "No Mask"):
                # pygame.mixer.music.play(0)
                counter_masks['no_mask'] += 1
            else :
                counter_masks['has_mask'] += 1

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


        text1 = "{}: {:}".format("People with MASKS: ", counter_masks['has_mask'])
        text2 = "{}: {:}".format("People with NO MASKS: ", counter_masks['no_mask'])

        cv2.putText(frame, text1, (10 , 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, text2, (10 , 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


        # mengkodekan bingkai mentah OpenCV ke jpg dan menampilkannya
        ret, jpeg = cv2.imencode('.jpg', frame)
        return (jpeg.tobytes(), counter_masks)








	
# def detect(self, frame, faceNet, maskNet):
#     # ambil dimensi bingkai dan kemudian buat gumpalan darinya
#         (h, w) = frame.shape[:2]
#         blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
#         (104.0, 177.0, 123.0))

#         # melewati gumpalan melalui jaringan dan mendapatkan deteksi wajah
#         faceNet.setInput(blob)
#         detections = faceNet.forward()

#         # inisialisasi daftar wajah kami, lokasi yang sesuai, dan daftar prediksi dari jaringan masker wajah kami
#         faces = []
#         locs = []
#         preds = []

#         # loop over the detections
#         for i in range(0, detections.shape[2]):
#             # ekstrak confidence (yaitu, probability) yang terkait dengan deteksi
#             confidence = detections[0, 0, i, 2]

#             # menyaring deteksi lemah dengan memastikan kepercayaan lebih besar dari kepercayaan minimum
#             if confidence > 0.5:
#                 # hitung (x, y)-koordinat kotak pembatas untuk objek
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")

#                 # pastikan kotak pembatas berada dalam dimensi bingkai
#                 (startX, startY) = (max(0, startX), max(0, startY))
#                 (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

#                 # ekstrak ROI wajah, ubah dari BGR ke pemesanan saluran RGB, ubah ukurannya menjadi 224x224, dan praproses
#                 face = frame[startY:endY, startX:endX]
#                 if face.any():
#                     face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#                     face = cv2.resize(face, (224, 224))
#                     face = img_to_array(face)
#                     face = preprocess_input(face)

#                     # tambahkan wajah dan lokasi ke variabel array 
#                     faces.append(face)
#                     locs.append((startX, startY, endX, endY))

#         # hanya membuat prediksi jika setidaknya satu wajah terdeteksi
#         if len(faces) > 0:
#             # untuk inferensi yang lebih cepat, kami akan membuat prediksi batch pada *semua* wajah secara bersamaan, bukan prediksi satu per satu dalam loop `untuk` di atas
#             faces = np.array(faces, dtype="float32")
#             preds = maskNet.predict(faces, batch_size=32)

#         # tentukan label kelas dan warna yang akan kita gunakan untuk menggambar kotak pembatas dan teks
#         return (locs, preds)
