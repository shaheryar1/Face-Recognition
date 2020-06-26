from sklearn.model_selection import train_test_split
# from PIL.Image import Image
import cv2
import os
import io
from PIL import Image
import numpy as np
from FaceRecognition import FaceRecogniton
from fastapi import FastAPI,File, UploadFile
import os

import base64


app = FastAPI()
def encode(img):
    pil_img = Image.fromarray(img)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return  new_image_string

@app.post("/add")
async def add(folder_path: str,person_name:str):
    try:
        if(os.path.exists(folder_path)):

            f = FaceRecogniton()
            encodings, labels = f.loadEncodings()
            if (person_name in labels):
                f.deletePerson(person_name)
            f.addPerson(folder_path,person_name)
            return {"Message":"Person Added"}

        else:
            raise  Exception("Path does not exists")
    except Exception as e:
        return {"Error":e}

@app.post("/recognize")
def recognize(file: bytes = File(...)):
    f=FaceRecogniton()
    image = Image.open(io.BytesIO(file)).convert("RGB")
    img = np.array(image)
    encodings, labels = f.loadEncodings()
    f.trainKNN(encodings,encodings,labels,labels,2)
    model=f.loadModel(type="knn")
    boxes,predictions,scores = f.inference_image(model, img)
    print(boxes)

    response={}
    response["faces"]=[]
    for box,label,s in zip(boxes,predictions,scores):
        temp={}
        temp["box"]=box
        temp["label"]=label
        temp["score"] = str(s)
        response["faces"].append(temp)

    response["image"]=encode(f.visualize(img,boxes,predictions))
    return response

@app.post("/del")
def delete(person_name:str):
    try:

        f = FaceRecogniton()
        encodings, labels = f.loadEncodings()


        if(person_name in labels)==False:
            return {"Error":"Person not found in database"}

        f.deletePerson(person_name)
        return {"Message":"Person deleted"}

    except Exception as e:
        return {"Error":e}
# 
#
#
# if __name__ == '__main__':
#     f=FaceRecogniton()
#
#     encodings,labels=f.loadEncodings()
#     print(labels)
#     type(encodings)
#     # X_train, X_test, y_train, y_test = train_test_split(encodings, labels, test_size=0.25, shuffle=True, random_state=142)
#     # print(len(X_train),len(X_test))
#     model =f.trainKNN(encodings,encodings,labels,labels,2)
#     f.inference_video('Tom.mp4',model)
#
#     # for path in os.listdir('test'):
#     #     img = cv2.imread('test/'+path)
#     #     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     #     img=f.inference_image(model,img)
#     #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     #     print(img.shape,path)
#     #     cv2.imwrite('results/'+path,img)