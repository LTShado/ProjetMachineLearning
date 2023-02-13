import os
import shutil
import requests

states = ["brésil", "france", "portugal"]

for state in states:
    path = os.path.join("dataset", state)
    os.mkdir(path)
    file_urls = open("Lib/urls/" + state + "_urls.txt", 'r')
    urls = file_urls.read().splitlines()

    count = 0
    for url in urls:
        try:
            res = requests.get(url, stream=True)
        except Exception:
            pass
        if res.status_code == 200:
            try:
                with open(path + "/" + str(count) + ".jpg", 'wb') as f:
                    shutil.copyfileobj(res.raw, f)
                    count += 1
            except Exception:
                pass
