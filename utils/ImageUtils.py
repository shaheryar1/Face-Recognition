import cv2,glob

import numpy as np

def drawBox(frame, bounding_box, name, resize_factor=1):
    '''
    :param frame: image as numpy array
    :param bounding_box: ( top, right, bottom, left)
    :param name: person name
    :param resize_factor:
    :return:
    '''
    rescaling = 1 / resize_factor;
    top, right, bottom, left = bounding_box
    top *= int(rescaling)
    right *= int(rescaling)
    bottom *= int(rescaling)
    left *= int(rescaling)
    padding = 20
    top += padding
    right += padding
    bottom += padding
    left -= padding
    # frame = cv2.resize(frame, (0, 0), fx=rescaling, fy=rescaling)
    frame = cv2.rectangle(frame, (right, top), (left, bottom), (0, 255, 255), 2)
    # Draw a label with a name
    # below the face
    frame = cv2.rectangle(frame, (left, bottom - 30), (right , bottom), (0, 255, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    frame = cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 1)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame