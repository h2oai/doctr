import glob
import json
from tqdm import tqdm
import cv2
image_list = glob.glob("./mnt/**/*.jpg", recursive = True)
labels_dict = {}
for image in tqdm(image_list):
    with open(image, 'rb') as f:
        check_chars = f.read()[-2:]
    if check_chars != b'\xff\xd9':
        print('Not complete image')
        continue
    img = cv2.imread(image)
    if img is None:
        print("empty image")
        continue
    if (img.shape[0] == 0) or (img.shape[1] == 0) or (img.shape[2] == 0):
        print("image with zero dimensions")
        continue
    labels_dict[image.split("mnt/images/")[-1]] = image.split("_")[1]
with open("./mnt/labels.json", "w") as f:
    json.dump(labels_dict, f)