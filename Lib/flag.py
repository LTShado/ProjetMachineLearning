from PMC import *
from PIL import Image
import glob

path_dataset = "Lib/dataset/"
path_Brasil = "bresil/"
path_France = "france/"
path_Portugal = "portugal/"

nbIm_Brasil = 447
nbIm_France = 597
nbIm_Portugal = 388

def getAllImagesPath():
    paths = []

    for file in glob.glob(path_dataset+"**/*.jpg"):
        file = file.replace('\\','/')
        paths.append(file)
        #print (file)
    return paths
def flagClassification(lib):
    print("start flagClassification")

    nbImInTraining = 20

    path_allImages = getAllImagesPath()
    imagesForTraining = np.random.choice(path_allImages,nbImInTraining)

    flagImages = []
    flagCountries = []

    for file  in imagesForTraining:
        #print(file)
        im = Image.open(file)
        im = im.resize((16,16),resample=1)

        im = list(im.getdata())
        im = [x for sets in im for x in sets]
        
        #print(im)
        flagImages.append(im)
        #print(flagImages[0])

        if path_Brasil in file:
            flagCountries.append([1,0,0])
        elif path_France in file:
            flagCountries.append([0,1,0])
        elif path_Portugal in file:
            flagCountries.append([0,0,1])

    flagImages = np.array(flagImages)
    flagCountries = np.array(flagCountries)

    print(len(flagImages[0]))
    #print(flagCountries[0])
    
    model = createModelPMC(lib,[768,100,100,100,3])
    trainPMC(lib, model, flagImages, flagCountries, True, 0.1, 1000)
    saveModelPMC(lib,model, "flagClassif.txt")
    print("end flagClassification")

def imageRecognition(lib,model, filename):
    im = Image.open("Lib/dataset/testFlag/"+filename)
    im.show()
    im = im.resize((16,16),resample=1)
    im.show()

    im = list(im.getdata())
    im = [x for sets in im for x in sets]

    p = np.array(im)
    print(len(p),p[0])
    pred = predictPMC(lib,model, p, True)
    res =""
    m = max(pred[0:3])
    if pred[0]>=0 and pred[0]==m:
        res = "Brasil"
    elif pred[1]>=0 and pred[1]==m:
        res = "France"
    elif pred[2]>=0 and pred[2]==m:
        res = "Portugal"

    print("I think this flag is for: ",res)

if __name__ == "__main__":
    lib = cdll.LoadLibrary(PATH_TO_SHARED_LIBRARY)

    flagClassification(lib)
    loadedModel = loadModelPMC(lib,"flagClassif.txt")
    imageRecognition(lib,loadedModel,"frenchflag.jpg")