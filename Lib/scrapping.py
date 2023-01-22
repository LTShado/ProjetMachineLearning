import os
import shutil
import requests

file_urls = open("./Lib/tomato_urls.txt", 'r')
urls = file_urls.read().splitlines()

count = 0
for url in urls:
    try:
        res = requests.get(url, stream=True)
    except Exception:
        pass
    if res.status_code == 200:
        count += 1
        try:
            with open("content/" + str(count) + ".jpg", 'wb') as f:
                shutil.copyfileobj(res.raw, f)
        except Exception:
            pass
