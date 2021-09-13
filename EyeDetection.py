#imports
import cv2
import numpy as np
import dlib
import pyautogui as pag
import keyboard

#sets up webcam capture, and also face detecto
cap = cv2.VideoCapture(0);
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#sets up font and x and y variables
font = cv2.FONT_HERSHEY_COMPLEX
oldX = 0
oldY = 0

#continually loops forever
while True:
    #sets up current frame of continuous webcam capture
    _, frame = cap.read()

    #converts current frame to grayscale for easier detection
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #when given a picture, the detector returns an arraylist of coordinate objects that point towards the different features the face
    faces = detector(gray)

    #loops through each point in the faces arraylist and draws a rectangle using the top, left, right, and bottom points around the face.
    for face in faces:
        x,y = face.left(),face.top()
        x1,y1 = face.right(),face.bottom()
        cv2.rectangle(frame, (x,y), (x1,y1), (0,0,255), 4)

        #uses the predictor function to determine x and y locations of facial landmarks in an arraylist
        landmarks = predictor(gray, face)

        #the left eye region is highlighted using markers 36-41 in the landmarks arraylist and put into a numpy array, which is the same thing as a Java arraylist
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)],np.int32)

        #using the arraylist we just made, a polygon is drawn around the left eye
        cv2.polylines(frame,[left_eye_region],True,(255,0,9),2)

        #tracks a rectangle around the eye region
        min_x = np.min(left_eye_region[:,0])
        max_x = np.max(left_eye_region[:,0])
        min_y = np.min(left_eye_region[:,1])
        max_y = np.max(left_eye_region[:,1])

        #takes new x and y of the eye
        newX = landmarks.part(36).x
        newY = landmarks.part(36).y

        #if the f key is pressed and the x and y have changed, the cursor is moved in the corresponding direction
        if(keyboard.is_pressed('f')):
            if(newX > oldX+10):
                pag.moveRel(-50,0,duration = 0.05)
            elif(newX < oldX-10):
                pag.moveRel(50,0,duration = 0.05)
            if(newY > oldY+10):
                pag.moveRel(0,50,duration = 0.05)
            elif(newY < oldY-10):
                pag.moveRel(0,-50,duration = 0.05)

        oldX = newX
        oldY = newY
    #if the v key is clicked, the mouse clicks
    if(keyboard.is_pressed('v')):
        pag.click()
    #displays the webcam capture for debugging purposes
    cv2.imshow("Camera",frame)

    #if the esc key is pressed, exit the program
    key = cv2.waitKey(1)
    if key == 27:
        break

#closes all windows and webcam capture
cap.release()
cv2.destroyAllWindows()