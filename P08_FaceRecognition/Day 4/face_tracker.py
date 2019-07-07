import cv2

face_cascade = cv2.CascadeClassifier('../day 2/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
frame_num = 1
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if frame_num == 1:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            tracker = cv2.TrackerCSRT_create()
            tracker.init(img, (x, y, w, h))
    (success, box) = tracker.update(img)

    # check to see if the tracking was a success
    if success:
        frame_num = frame_num + 1
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(img, (x, y), (x + w, y + h),
                      (255, 0, 0), 2)

    cv2.imshow('img', cv2.flip(img, 1))
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
