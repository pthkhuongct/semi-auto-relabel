import cv2
import os
import numpy as np

drawing = False
ix, iy = -1, -1
boxes = []
current_class = 0

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, boxes, current_class

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.putText(img_copy, str(current_class), (ix, iy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow('Labeled Image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.putText(img, str(current_class), (ix, iy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        boxes.append((ix, iy, x, y, current_class))

def load_model(weight_path, cfg_path):
    net = cv2.dnn.readNet(weight_path, cfg_path)
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def auto_label(image_path, output_path, net, output_layers, confidence_threshold=0.01):
    global img, boxes, current_class
    boxes = []  # Reset boxes for each image

    # Load image
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # Detect objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to display on the label
    class_ids = []
    confidences = []
    detected_boxes = []

    # For each detection from each output layer get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < confidence_threshold)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                detected_boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to eliminate redundant overlapping boxes with
    # lower confidences
    indexes = cv2.dnn.NMSBoxes(detected_boxes, confidences, confidence_threshold, 0.4)

    for i in range(len(detected_boxes)):
        if i in indexes:
            x, y, w, h = detected_boxes[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, str(class_ids[i]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            boxes.append((x, y, x + w, y + h, class_ids[i]))

    cv2.namedWindow('Labeled Image')
    cv2.setMouseCallback('Labeled Image', draw_rectangle)

    while True:
        img_copy = img.copy()
        cv2.putText(img_copy, f"Current class: {current_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Labeled Image', img_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to move to next image
            break
        elif key == ord('0'):  # Set class to 0
            current_class = 0
            print("Current class set to 0")
        elif key == ord('1'):  # Set class to 1
            current_class = 1
            print("Current class set to 1")
        elif key == ord('2'):  # Set class to 2
            current_class = 2
            print("Current class set to 2")

    with open(output_path, 'w') as f:
        for box in boxes:
            x1, y1, x2, y2, label = box
            x_center = (x1 + x2) / 2.0 / width
            y_center = (y1 + y2) / 2.0 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            f.write(f"{label} {x_center} {y_center} {w} {h}\n")

    cv2.destroyAllWindows()

# Example usage
image_folder = 'images/'
label_folder = 'label/'
if not os.path.exists(label_folder):
    os.makedirs(label_folder)

# Using only YOLOv4 model
model = {"weights": "darknet/yolov4-5000.weights", "cfg": "darknet/yolov4.cfg"}

net, output_layers = load_model(model["weights"], model["cfg"])

for image_name in os.listdir(image_folder):
    if image_name.endswith(('.jpg', '.png', '.jpeg')):  # Chỉ xử lý các định dạng ảnh phổ biến
        image_path = os.path.join(image_folder, image_name)
        label_path = os.path.join(label_folder, os.path.splitext(image_name)[0] + '.txt')
        auto_label(image_path, label_path, net, output_layers)
