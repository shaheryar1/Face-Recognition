import cv2
import numpy as np
import face_recognition



img = cv2.imread(r'C:\Users\Shaheryar\PycharmProjects\Face-Recognition\test\b.jpg')
known_person=cv2.imread(r'C:\Users\Shaheryar\PycharmProjects\Face-Recognition\dataset\Tom Hank\https___cdn.cnn.com_cnnnext_dam_assets_200103175454-tom-hanks.jpg')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
locations = face_recognition.face_locations(known_person,model="cnn")
print(len(locations))
known_person_encodings=face_recognition.face_encodings(known_person,known_face_locations=locations)

print(np.array(known_person_encodings).shape)

# for location in locations:
#     print(location)
#     i=i+1
#     y1,x1,y2,x2=location
#     cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),1)
#
#     crop=img[y1:y2,x2:x1]
#     print(crop.shape)
#     cv2.imshow(str(i),crop)
# cv2.waitKey(30000)





# if __name__ == '__main__':
#     cap=cv2.VideoCapture('Tom.mp4')
#
#
#     # Define the codec and create VideoWriter object
#     fps = cap.get(5)
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 360))
#     i = 0
#     while True:
#         ret,frame=cap.read()
#         print(frame.shape)
#         if (frame is None):
#             break
#         i=i+1
#         font = cv2.FONT_HERSHEY_DUPLEX
#         frame = cv2.putText(frame,"Tom  Hanks", (50 , 50), font, 1, (255, 255, 0), 1)
#         frame = cv2.rectangle(frame,(100,100),(200,200),(255,255,0),2)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         cv2.imshow('Tom Hanks',frame)
#         out.write(frame)
#         # cv2.waitKey(30)
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#
#
#
#     print(i)