import os
from PIL import Image

states = ["france", "usa", "coree"]

for state in states:
    count = 1
    for filename in os.listdir("dataset/" + state):
        f = os.path.join("dataset/" + state, filename)
        if os.path.isfile(f):
            try:
                image = Image.open(f)
                res = image.resize((120, 80))
                res.save(f)
            except Exception:
                pass
