import requests
from utils import process_images
from PIL import Image

def test():
    print("hello")

imgstr = process_images.img2str(Image.open(".images/fd7151edcd48ec1b073b0bd23f3c8810.png").convert("RGB")
)[0]

data_to_send = {
    "text": "a big dog",
    #"image": imgstr,
    #"strength": 0.9,
    "num_steps": 20,
    "guidance_scale": 7.5}

url='http://0.0.0.0:8000/t2i'


response = requests.post(url, json=data_to_send).json()
print(response.keys())
print(response["time"])

imgs = process_images.str2img(response["imgs"])

imgs[0].show()

