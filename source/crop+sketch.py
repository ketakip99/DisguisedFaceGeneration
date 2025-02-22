import cv2
import glob
import matplotlib.pyplot as plt
import os


folder_dir = "D:/Rajasi/Ketaki/scrapped_data/*.*"
src_path = "D:/Rajasi/Ketaki/scrapped_data"

list_with_ext = os.listdir(src_path)

list_wo_ext = []
for name in list_with_ext:
   base, extension = os.path.splitext(name)
   list_wo_ext.append(base)

dest_path ="D:/Rajasi/Ketaki/BARC_sketches/"
dest_path2 ="D:/Rajasi/Ketaki/BARC_cropped/"
i = 0
   
for file in glob.glob(folder_dir):
    image = cv2.imread(file)
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    face_classifier = cv2.CascadeClassifier("D:/Rajasi/code/haarcascade_frontalface_default.xml")
   
    face = face_classifier.detectMultiScale(grey_img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
   
    for (x, y, w, h) in face:
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cropped_image = image[y:y+h, x:x+w]
   
    new_name1 = list_wo_ext[i] + "_crop.png"
    cv2.imwrite(os.path.join(dest_path2 , new_name1), cropped_image)
   
    grey_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(grey_img)
    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    invertedblur = cv2.bitwise_not(blur)
    sketch = cv2.divide(grey_img, invertedblur, scale=256.0)
    sharpened_image2 = cv2.Laplacian(sketch, cv2.CV_64F)
    new_name = list_wo_ext[i] + "_sketch.png"
    cv2.imwrite(os.path.join(dest_path , new_name), sketch)
    i = i+1
   
print("Conversion successful!")