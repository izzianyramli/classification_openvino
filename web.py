import flask
import redis
from PIL import Image
import numpy as np
import settings
from helpers import base64_encode_image
import json
import uuid
import io
import cv2
import time

app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def process_image(image, target):
    # image.show()

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = (np.array(image) - 0) / 255.0
    image = image.transpose((2, 0, 1))
    image = image.reshape(1, 3, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH)  # reconfirm the batch size value

    return image


@app.route("/")
def home():
    return ("Hi")


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = process_image(image, (settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH))
            image = image.copy(order="C")

            k = str(uuid.uuid4())
            d = {"id": k, "image": base64_encode_image(image)}

            db.rpush(settings.IMAGE_QUEUE, json.dumps(d))

            while True:
                output = db.get(k)

                if output is not None:
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    db.delete(k)
                    break
                time.sleep(settings.CLIENT_SLEEP)

            data["success"] = True

        else:
            flask.jsonify({'error': 'no file'}), 400

    return flask.jsonify(data)


if __name__ == "__main__":
    print("\t[INFO] Starting web service\n")
    app.run()