from threading import Thread
import requests
import argparse
import time
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                help="Required. Path to input image.", required=True, type=str)
ap.add_argument("-c", "--count",
                help="Required. Number of requests.", required=True, type=int)
args = vars(ap.parse_args())

URL = "http://localhost:5000/predict"
IMAGE_PATH = args["image"]

NUM_REQUESTS = args["count"]
SLEEP_COUNT = 0.05

# start_time = datetime.now()
# # print("Start time:  ", start_time)
thread_list = []


def call_predict_endpoint(n):
    image = open(IMAGE_PATH, "rb").read()
    payload = {"image": image}

    r = requests.post(URL, files=payload).json()

    if r["success"]:
        print("[INFO] thread {} OK".format(n))
    else:
        print("[INFO] thread {} FAILED".format(n))

# loop over the number of threads
for i in range(0, NUM_REQUESTS):
    # start a new thread to call the API
    t = Thread(target=call_predict_endpoint, args=(i,))
    thread_list.append(t)
    # t.daemon = True
    # t.start()
    # time.sleep(SLEEP_COUNT)
start_time = datetime.now()
for thread in thread_list:
    thread.start()

for thread in thread_list:
    thread.join()

# t.join()
# time.sleep(300)

total_time = (datetime.now() - start_time).total_seconds()
print("threading complete.")

# total_time = (((datetime.now() - start_time).total_seconds()) -
#               (SLEEP_COUNT*NUM_REQUESTS) - 300)
# print("\t Time interval:{} \n".format(total_time))
rps = NUM_REQUESTS / total_time
print("\n\t Request per second: {}".format(rps))
print("\n\t Total time taken to infer {} requests: {}".format(
    NUM_REQUESTS, total_time))
print("\n\t Average of {} requests: {}".format(
    NUM_REQUESTS, total_time/NUM_REQUESTS))
