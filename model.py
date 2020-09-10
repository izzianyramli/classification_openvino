# from openvino import inference_engine as ie
from openvino.inference_engine import IECore, IENetwork
import numpy as np
from PIL import Image
import settings
from helpers import base64_decode_image
import redis
import time
import json
import os
import platform
import argparse
from datetime import datetime
import psutil

db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def classify_process():
    ie = IECore()
    exec_net = ie.load_network(network=net, device_name=device)
    print(BATCH_SIZE)
    print(device)
    print("\t[INFO] Model loaded\n")

    # cpu_after = psutil.cpu_percent()
    # print(f"Total CPU usage after load model: {cpu_after}%")
    # print(f"CPU usage different: {cpu_after - cpu_before}%")

    while True:
        queue = db.lrange(settings.IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        imageIDs = []
        batch = None

        # define number of batch size -try to send multiple image. refer to keras original model (we can define the number of queue)
        for q in queue:
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(
                q["image"], settings.IMAGE_DTYPE,
                # (1,h,w,c))
                (1, c, h, w))  # reconfirm the batch size value

            if batch is None:
                batch = image
            else:
                batch = np.vstack([batch, image])
            imageIDs.append(q["id"])

        if len(imageIDs) == BATCH_SIZE:
            # print("\t[INFO] No. of images in a batch: {}".format(len(imageIDs)))
            # print("\t[INFO] Batch size: {}".format(batch.shape))
            start_time = datetime.now()
            res = exec_net.infer(inputs={input_blob: batch})
            # print(image.shape)
            res = res[out_blob]
            # print(res)

            if labels:
                with open(labels, 'r') as f:
                    labels_map = [
                        x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
            else:
                labels_map = None

            for imageID in imageIDs:
                output = []

                for i, probs in enumerate(res):
                    probs = np.squeeze(probs)
                    top_ind = np.argsort(
                        probs)[-5:][::-1]

                    for id in top_ind:
                        det_label = labels_map[id] if labels_map else "{}".format(
                            id)
                        r = {"label": det_label,
                             "probability": float(probs[id])}
                        output.append(r)

                db.set(imageID, json.dumps(output))
            db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)
            duration = (datetime.now() - start_time).total_seconds()
            throughput = BATCH_SIZE/duration
            print(duration)
            # print("duration: {}".format(duration))
            # print("\t[INFO] Duration: {} seconds".format(duration))
            # print("\t[INFO] Throughput: {} fps".format(throughput))
        # else:
            # print("\t[INFO] Number of image(s) in batch is {}. Please input {} image(s).".format(
            # len(imageIDs), BATCH_SIZE))

        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model",
                    help="Required. Path to the trained model file.", required=True, type=str)
    ap.add_argument("-l", "--labels",
                    help="Optional. Labels mapping file.", type=str)
    ap.add_argument("-d", "--data_type",
                    help="Optional. Data type FP16/FP32. Default FP16.", default="FP16", type=str)
    ap.add_argument("-o", "--output_dir",
                    help="Optional. Path to save the optimized model (.xml and .bin file).", type=str)
    ap.add_argument("--device", default="CPU", type=str,
                    help="Optional. Device CPU or GPU. Default is CPU.")
    ap.add_argument("-b", "--batch_size",
                    help="Optional. Maximum number of image can be process.", type=int)
    args = vars(ap.parse_args())

    is_win = "windows" in platform.platform().lower()
    if is_win:
        mo_path = '"C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo.py"'
    else:
        mo_path = '/opt/intel/openvino/deployment_tools/model_optimizer/mo.py'

    device = args["device"]
    orig_model = args["model"]
    output_dir = args["output_dir"]
    data_type = args["data_type"]
    labels = args["labels"]
    # maximum number of images that can be process for stress test.
    BATCH_SIZE = args["batch_size"]

    # if orig_model is not .xml file, convert using model optimizer. if-else statement
    ext = os.path.splitext(orig_model)[-1].lower()

    if ext == ".xml":
        model_xml = orig_model
    else:
        model_name, extension = os.path.splitext(os.path.basename(orig_model))
        input_shape = [BATCH_SIZE, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3]
        input_shape_str = str(input_shape).replace(' ', '')

        cmd = "python {} --input_model {} --output_dir {} --input_shape {} --data_type {}".format(
            mo_path, orig_model, output_dir, input_shape_str, data_type)
        os.system(cmd)
        model_xml = os.path.splitext(
            output_dir)[0] + ("\\") + model_name + ".xml"

    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    print("\t[INFO] Loading network files: \n\t{}\n\t{}\n".format(
        model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    b, c, h, w = net.inputs[input_blob].shape
    BATCH_SIZE = b
    print("{}, {}, {}, {}".format(b, c, h, w))

    # cpu_before = psutil.cpu_percent()
    # print(f"Total CPU usage before load model: {cpu_before}%")
    classify_process()
