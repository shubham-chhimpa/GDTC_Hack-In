import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer\\trainningData.yml')
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX
names = ["", "Shubham", "Vishal", "Vinit"]
cap = cv2.VideoCapture(0)
frame_num = 1
faceCascade = cv2.CascadeClassifier('../day 2/haarcascade_frontalface_default.xml')

while True:
    ret, image = cap.read()
    # Create the haar cascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if frame_num == 1:

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        print("Found {0} faces!".format(len(faces)))

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            tracker = cv2.TrackerCSRT_create()
            tracker.init(image, (x, y, w, h))

    (success, box) = tracker.update(image)
    # check to see if the tracking was a success
    if success:
        frame_num = frame_num + 1
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(image, (x, y), (x + w, y + h),
                      (255, 0, 0), 2)
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        print(str(id) + "  matched")

        cv2.putText(image, names[int(id)], (x, y + h + 20), font, .6, (0, 255, 0), 2)

    cv2.imshow("Faces found", image)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
