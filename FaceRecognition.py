
import pickle
import face_recognition
from utils.ImageUtils import drawBox
from datetime import datetime
import face_recognition
from sklearn import svm
import os
import  numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from imutils import face_utils
import dlib
import cv2
import math
import time
from sklearn import neighbors

class FaceRecogniton:

    def __init__(self):
        self.encodings_pkl_file_name = "encodings.pkl"
        self.knn_model_file ='models/knn.pkl'
        self.svm_model_file = 'models/svm.pkl'

    def saveEncodings(self, encodings, labels):
        dic = {
            "encodings": encodings,
            "labels": labels,
        }

        f = open(self.encodings_pkl_file_name, "wb")
        pickle.dump(dic, f)
        f.close()

    def loadEncodings(self):
        with open(self.encodings_pkl_file_name, 'rb') as file:
            dict = pickle.load(file)

        return dict["encodings"], dict["labels"];

        # Overloaded method

    def readImagesFromFolder(self, folder_path):
        total_images = [];

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)



            img = cv2.imread(img_path)
            # img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB);
            total_images.append(img)

        return total_images

    def getFaceEncodings(self, faces, bounding_boxes=None):
        '''
        :param faces: list of images having faces
        :param bounding_boxes: list of bounding boxes
        :return: list of encodings
        '''
        encodings = []
        for i in range(0, len(faces)):
            multi_channel = faces[i]

            if bounding_boxes is not None:
                face_enc = face_recognition.face_encodings(multi_channel, known_face_locations=[bounding_boxes[i]])[
                    0]
            else:
                face_enc = face_recognition.face_encodings(multi_channel)[0]

            if (i % 20 == 0):
                print("**Calculating encodings**")

            encodings.append(face_enc)

        return encodings;

    def isknown(self, known_encodings, test_encoding):
        '''
        :param known_encodings: encodings from training data
        :param test_encoding: test image encodings
        :return: True => Known face  OR  False => Unknown
        '''
        face_distances = face_recognition.face_distance(known_encodings, test_encoding)
        result = False
        for i, face_distance in enumerate(face_distances):
            if face_distance < 0.55:
                # print(face_distance)
                result = True
                return result,face_distance
        return result,face_distance

    def deleteEncodings(self, target_label):

        encodings, labels = self.loadEncodings()
        encodings = np.array(encodings)
        labels = np.array(labels)
        print("deleting")
        idx = np.where(labels == target_label)
        print(len(idx[0]))
        labels = np.delete(labels, idx)
        encodings = np.delete(encodings, idx, axis=0)

        return list(encodings), labels;

    def addEncoding(self, new_label, new_images):
        encodings, labels = self.loadEncodings()
        labels = np.array(labels)
        if (len(np.where(labels == new_label)[0]) > 0):
            print("Already exist")
            return encodings, labels;
        # new_enc=self.getFaceEncodings(images)

        for i in range(len(new_images)):
            face_locations = face_recognition.face_locations(new_images[i], model='hog')
            if (len(face_locations) > 0):
                face_encodings = face_recognition.face_encodings(new_images[i], face_locations)[0]
                labels = np.append(labels, new_label)
                encodings.append(face_encodings)
        print(len(labels), "encodings added !")
        return encodings, labels;

    def addPerson(self, folder_path,label):
        imgs = self.readImagesFromFolder(folder_path)
        print(len(imgs))
        new_encodings, new_labels = self.addEncoding(label, imgs)
        self.saveEncodings(new_encodings, new_labels)



    def trainKNN(self, X_train, X_test, y_train, y_test, K):
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=K, weights='distance')
        knn_clf.fit(X_train, y_train)

        if self.knn_model_file is not None:
            with open(self.knn_model_file, 'wb') as f:
                pickle.dump(knn_clf, f)

        p = knn_clf.predict(X_train)
        # print("Train Accuracy on KNN :", accuracy_score(y_train, p) * 100)

        p = knn_clf.predict(X_test)
        # print("Test Accuracy on KNN :", accuracy_score(y_test, p) * 100)

        return knn_clf;

    def trainSVM(self, X_train, X_test, y_train, y_test, C):
        clf = svm.SVC(C=C,probability=True)
        clf.fit(X_train, y_train)

        if self.knn_model_file is not None:
            with open(self.svm_model_file, 'wb') as f:
                pickle.dump(clf, f)

        p = clf.predict(X_train)
        print("Train Accuracy on SVM :", accuracy_score(y_train, p) * 100)

        p = clf.predict(X_test)
        print("Test Accuracy on SVM :", accuracy_score(y_test, p) * 100)

        return clf;

    def loadModel(self,type="knn"):
        if type=="knn":
            with open(self.knn_model_file, 'rb') as file:
                model = pickle.load(file)
        else:
            with open(self.svm_model_file, 'rb') as file:
                model = pickle.load(file)

        return model

    def inference_image(self,model,img):
        known_encodings, labels = self.loadEncodings()
        predictions=[]
        scores=[]
        face_locations = face_recognition.face_locations(img, model='hog')
        if (len(face_locations) > 0):
            face_encodings = face_recognition.face_encodings(img, face_locations)
            for face_encoding,face_location in zip (face_encodings,face_locations):
                p = model.predict([face_encoding])

                flag,distance=self.isknown(known_encodings, face_encoding)
                if (flag):
                    predictions.append(p[0])
                else:
                    predictions.append("Unknown")
                scores.append(distance)

        return face_locations,predictions,scores
    def visualize(self,img,face_locations,predictions):
        for face_location,label in zip(face_locations,predictions):
            img = drawBox(img, face_location, label)
        return img



    def deletePerson(self, target_label):
        new_encodings, new_labels = self.deleteEncodings(target_label)
        self.saveEncodings(new_encodings, new_labels)

    def inference_video(self,video_source,model):
        cap = cv2.VideoCapture(video_source)
        i = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if (i % 60 == 0):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes,labels=self.inference_image(model,frame)
                img=self.visualize(frame,boxes,labels)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)



            i=i+1
            cv2.imshow('frame', img)
            # cv2.imshow('frame', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break





