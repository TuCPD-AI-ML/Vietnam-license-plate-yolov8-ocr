from ultralytics import YOLO
import matplotlib.pyplot as plt 
import cv2
import numpy as np
import easyocr
import os

model = YOLO('Plate_Recognition/recognition/best2.pt')
model.fuse()
reader = easyocr.Reader(['en'])
plt.figure()

folder_path = 'Plate_Recognition/recognition/test'
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for i, image_file in enumerate(image_files):
    image_path = os.path.join(folder_path, image_file)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    cv2.imshow('Image', img)

    result = model(source=img, conf=0.3)
    # print(result)
    for r in result:
        box = r.boxes.cpu().numpy()
    xyxy = box.xyxy
    # print(xyxy)
    if len(xyxy) > 0:
        x, y, x_max, y_max = list(map(int, xyxy[0][:4]))
        plate = img[y:y_max, x:x_max].copy()
        cv2.imshow('Plate', plate)

        plate_gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
        plate_blur = cv2.GaussianBlur(plate_gray, (5, 5), 0)
        edges = cv2.Canny(plate_blur, 100, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        # test_img1 = plate.copy()
        # cv2.drawContours(test_img1, max_contour, -1, (0, 255, 0), 10)
        # plt.imshow(test_img1)
        epsilon = 0.02*cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        if len(approx) == 4:
            width, height = 300, 100
            src = np.float32([list(approx[0][0]), list(approx[3][0]), list(approx[1][0]), list(approx[2][0])])
            dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

            M = cv2.getPerspectiveTransform(src, dst)

            test = plate_gray.copy()
            new_plate = cv2.warpPerspective(test, M, (width, height))

            _, thresholded_img = cv2.threshold(new_plate, 127, 255, cv2.THRESH_BINARY)
            dilation = cv2.dilate(thresholded_img, np.ones((4,4), np.uint8), iterations = 1)

            cv2.imshow(f'ảnh {i+1}', new_plate)
            cv2.imshow(f'ảnh {i+1}', thresholded_img)
            cv2.imshow(f'ảnh {i+1}', dilation)

            number = reader.readtext(plate_gray)
            if len(number) == 1:
                print(f'{image_file}:', number)    
                print(f'{image_file}:', number[0][1])
            else:
                print(f'{image_file}: No read number')
        else:
            number = reader.readtext(plate_gray)
            if len(number) == 1:
                print(f'{image_file}:', number)    
                print(f'{image_file}:', number[0][1])
            else:
                print(f'{image_file}: No read number')
    else:
        print(f'{image_file}: No Detection')
