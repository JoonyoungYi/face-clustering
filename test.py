import face_recognition
import cv2
import numpy as np
import scipy.cluster.vq

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

SCALE = 2
USER_NUMBER = 10
SKIPPING_FRAME = 29
VIDEO_FILENAME = 'youtube1.mov'
FACE_ENCODING_ARRAY_FILENAME = 'youtube1-face-encoding-array.npy'
# CENTROID_FILENAME = 'youtube1-centroid.npy'

# try:
#     centroid = np.load(CENTROID_FILENAME)
# except:
#     print('MISS!')
#     centroid = None

# if centroid is None:
try:
    face_encoding_array = np.load(FACE_ENCODING_ARRAY_FILENAME)
except:
    face_encoding_array = None

if face_encoding_array is None:
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(VIDEO_FILENAME)
    face_encoding_array = np.zeros((0, 128))

    while True:
        # print(len(known_face_encodings))

        # Grab a single frame of video
        _, frame = video_capture.read()
        if frame is None:
            break

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=1 / SCALE, fy=1 / SCALE)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        for face_encoding in face_encodings:
            face_encoding_array = np.append(
                face_encoding_array, face_encoding.reshape(1, 128), axis=0)
            print(face_encoding_array.shape)

        print('>> Frame Processing')

        for _ in range(SKIPPING_FRAME - 1):
            video_capture.grab()

    np.save(FACE_ENCODING_ARRAY_FILENAME, face_encoding_array)
    video_capture.release()

print('>> Clustering Images')
print(face_encoding_array.shape)

known_face_encodings = []
for i in range(face_encoding_array.shape[0]):
    if not known_face_encodings:
        known_face_encodings.append(face_encoding)
        continue

    face_encoding_array[i, :]

#     centroid, _ = scipy.cluster.vq.kmeans2(face_encoding_array, USER_NUMBER)
#     np.save(CENTROID_FILENAME, centroid)
#
# print(type(centroid))
# print(centroid.shape)
# known_face_encodings = [centroid[i, :] for i in range(centroid.shape[0])]
# print('>> Clustering Finished!')

# # Load a sample picture and learn how to recognize it.
# obama_image = face_recognition.load_image_file("obama.jpg")
# obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
# print(type(obama_face_encoding))
# assert False

# # Load a second sample picture and learn how to recognize it.
# biden_image = face_recognition.load_image_file("biden.jpg")
# biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
#
# # Create arrays of known face encodings and their names
# known_face_encodings = [obama_face_encoding, biden_face_encoding]
# known_face_names = ["Barack Obama", "Joe Biden"]

# Initialize some variables
face_locations = []
face_encodings = []

video_capture = cv2.VideoCapture(VIDEO_FILENAME)

while True:
    # Grab a single frame of video
    _, frame = video_capture.read()
    if frame is None:
        break

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=1 / SCALE, fy=1 / SCALE)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame,
                                                     face_locations)

    for _ in range(SKIPPING_FRAME - 1):
        video_capture.grab()

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        distances = face_recognition.face_distance(known_face_encodings,
                                                   face_encoding)

        # If a match was found in known_face_encodings, just use the first one.
        index = np.argmin(distances)
        name = "{}".format(index)

    # Display the results
    for top, right, bottom, left in face_locations:
        # for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= SCALE
        right *= SCALE
        bottom *= SCALE
        left *= SCALE

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255),
                      cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255,
                                                                     255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    for _ in range(SKIPPING_FRAME - 1):
        video_capture.grab()

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
