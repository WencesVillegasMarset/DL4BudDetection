import os
import gdown
import zipfile

DATA_URL = "https://drive.google.com/uc?id=1e4Vmknt5hWaWGSOD5kfxuq6w_QdNCnrn"
CHECKSUM = "d05ce2f4b81fbac0e434b3dbb6b7730a"
IMAGES_PATH = os.path.join(".", "images")
ZIP_PATH = os.path.join(".", "bud_dataset.zip")

if not os.path.exists(ZIP_PATH):
    url = DATA_URL
    output = "bud_dataset.zip"
    gdown.download(url, output, quiet=False)

    gdown.cached_download(url, output, md5=CHECKSUM)
    print("Dataset downloaded!")

if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as f:
        f.extractall(path=IMAGES_PATH)

print("Dataset extracted!")
