import requests
import cv2
import argparse
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                help="Required. Path to input image.", required=True, type=str)
args = vars(ap.parse_args())

URL = "http://127.0.0.1:5000/predict"
IMAGE_PATH = args["image"]

print("\t[INFO] Image file name: {}\n".format(IMAGE_PATH))
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

image = cv2.imread(IMAGE_PATH)

print("\t[INFO] Loading image to server\n")
start_time = datetime.now()
r = requests.post(URL, files=payload).json()
print("\t[INFO] Image passed to server\n")

if r["success"]:
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(
            i+1, result["label"], result["probability"]))
else:
    print("Request failed")

total_time = "{:.2f}".format((datetime.now() - start_time).total_seconds())
print("Time interval: {} seconds\n".format(total_time))