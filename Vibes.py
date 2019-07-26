import cv2
from scipy.spatial import distance
import dlib
import numpy as np
from imutils import face_utils
import random
import time

import speech_recognition as sr

face_landmark_path = './shape_predictor_68_face_landmarks.dat'

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]



def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
    
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(face_landmark_path)# crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(0)


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def main():
    # return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)
    flag_eye=0
    flag_face_down=0
    while cap.isOpened():
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (0,0), fx=1.5, fy=1.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape) # converting to NumPy Array
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < thresh:
                flag_eye += 1
                print (flag_eye)
                if flag_eye >= frame_check:
                    cv2.putText(frame, "**************************ALERT!*************************", (0, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "**************************ALERT!*************************", (0, 460),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # print ("Drowsy")
            else:
                flag_eye = 0

        if ret:
            face_rects = detector(frame, 0)

            if len(face_rects) > 0:
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle = get_head_pose(shape)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                for start, end in line_pairs:
                    cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

                cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 140), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)

                if (euler_angle[0, 0] > 7):
                    flag_face_down += 1
                    print (flag_face_down)
                    if flag_face_down >= frame_check:
                        cv2.putText(frame, "**************************ALERT!*************************", (0, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "**************************ALERT!*************************", (0, 460),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    flag_face_down = 0

            cv2.imshow("demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`.

    Returns a dictionary with three keys:
    "success": a boolean indicating whether or not the API request was
               successful
    "error":   `None` if no error occured, otherwise a string containing
               an error message if the API could not be reached or
               speech was unrecognizable
    "transcription": `None` if speech could not be transcribed,
               otherwise a string containing the transcribed text
    """
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response object accordingly
    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response


if __name__ == "__main__":
    # create recognizer and mic instances
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        while True:
            print('Input command : ')
            guess = recognize_speech_from_mic(recognizer, microphone)
            if guess["transcription"]:
                break
            if not guess["success"]:
                break
            print("I didn't catch that. What did you say?\n")

        # if there was an error, stop the game
        if guess["error"]:
            print("ERROR: {}".format(guess["error"]))
            break

        # show the user the transcription
        print("You said: {}".format(guess["transcription"]))
        # determine if guess is correct and if any attempts remain
        toCheck = guess["transcription"].lower()
        if toCheck == "quit":
            break 
        if toCheck == "start car":
            main()
