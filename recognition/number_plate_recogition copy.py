from ultralytics import YOLO
import matplotlib.pyplot as plt 
import cv2
import numpy as np
import easyocr
import os

model = YOLO('Plate_Recognition/recognition/best.pt')
model.fuse()
reader = easyocr.Reader(['en'])
plt.figure()

folder_path = 'Plate_Recognition/recognition/test'
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for i, image_file in enumerate(image_files):
    image_path = os.path.join(folder_path, image_file)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    # cv2.imshow('Image', img)

    result = model(source=img, conf=0.3)
    # print(result)
    print(f'{image_file}:')
    for r in result:
            box = r.boxes.cpu().numpy()
            xyxy = box.xyxy
            for plate_coords in xyxy:
                x, y, x_max, y_max = list(map(int, plate_coords[:4]))
                plate = img[y:y_max, x:x_max].copy()
            
                plate_gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
                result = reader.readtext(plate_gray)
                if result:
                    for detection in result:
                        text = detection[1]
                        print(text)
                else:
                    print("Không tìm thấy biển số xe.")
                
                print("\n")
