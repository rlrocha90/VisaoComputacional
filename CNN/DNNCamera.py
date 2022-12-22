import cv2

path_model = "../dnn_model/yolov4-tiny.cfg"
path_weight = "../dnn_model/yolov4-tiny.weights"
net = cv2.dnn.readNet(path_weight, path_model)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)
classes = []
with open("../dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


print(classes)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    ret, frame = cap.read()
    (class_ids, score, bboxes) = model.detect(frame)
    for class_ids, score, bboxes in zip(class_ids, score, bboxes):
        (x, y, w, h) = bboxes
        class_name = classes[class_ids]
        cv2.putText(frame, str(class_name), (x, y), 0, 1, (0, 255, 0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    print("class ids", class_ids)
    print("scores", score)
    print("bboxes", bboxes)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)