import requests


def download_dataset(url, file_path):
    with requests.get(url, stream=True) as req:
        with open(file_path, "wb") as f:
            for chunk in req.iter_content(4 * 1024):
                f.write(chunk)
