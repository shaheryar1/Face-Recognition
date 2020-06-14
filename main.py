from sklearn.model_selection import train_test_split
from PIL.Image import Image
import cv2
import os
from FaceRecognition import FaceRecogniton

if __name__ == '__main__':
    f=FaceRecogniton()

    encodings,labels=f.loadEncodings()
    print(labels)
    type(encodings)
    X_train, X_test, y_train, y_test = train_test_split(encodings, labels, test_size=0.25, shuffle=True, random_state=142)
    print(len(X_train),len(X_test))
    model =f.trainKNN(X_train,X_test,y_train,y_test,2)


    for path in os.listdir('test'):
        img = cv2.imread('test/'+path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=f.inference_image(model,img,X_train)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(img.shape,path)
        cv2.imwrite('results/'+path,img)