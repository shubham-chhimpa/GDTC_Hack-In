import cv2
import imutils


# Create the haar cascade
faceCascade = cv2.CascadeClassifier('../day 2/haarcascade_frontalface_default.xml')

id = input("input id of user to be detected")
sampleNum = 0
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    image = imutils.resize(image, height=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
        sampleNum = sampleNum + 1
        cv2.imwrite("dataset/user." + str(id) + "." + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.waitKey(100)

    cv2.imshow("Faces found", image)
    cv2.waitKey(1)
    if (sampleNum > 40):
        break
cv2.destroyAllWindows()