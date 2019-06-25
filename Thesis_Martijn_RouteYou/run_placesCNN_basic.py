# PlacesCNN for scene classification
#
# by Bolei Zhou
# by Martijn Vanallemeersch
# last modified by Martijn Vanallemeersch, Jun.06, 2019 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
import datetime
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import xlsxwriter
from PIL import Image
from anytree import Node, RenderTree
from collections import defaultdict
import  json


# importing csv module
import csv

# th architecture to use
arch = 'resnet18'
# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
#model_file = test
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    #weight_url = test
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

from os import listdir
from os.path import isfile, join
mypath = 'C:\\Users\\marti\\Documents\\Kuleuven\\Masterjaar\\Masterproef\\fotos-pieter\\fotos\\Belgie'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

GeneralTypeID_dict = {}
POITypes = dict()
Childs = dict()
Parent = dict()

RelatedWord = dict()

church = ["synagoge", "mausoleum","tower"]
castle = ["moat","canal","pond","palace","ruin","formal_garden"]
palace = ["moat","canal","pond","castle","ruin","formal_garden"]
bridge = ["canal","river","viaduct","aqueduct", "lock_chamber", "rope_bridge","moat","industrial_area"]
# street = ["crosswalk","plaza","alley"]
# crosswalk = ["street","plaza","alley"]
# park = ["forest_road","picnic_area","forest_pad","field/wild", "japanese_garden","botanical","garden/yard","rainforest","path/forest_path","vegetable_garden"]
# lawn = ["formal_garden","topiary_garden","botanical_garden","yard"]
# building = ["crosswalk","parking_garage","synagoge","hangar","farm","manufactured_home","burrough","patio","porch","museum"]
# house = ["oast_house"]
# #square = ["fountain"]
# hotel_room = ["bedchamber","bedroom","youth_hostel"]
# bedchamber = ["hotel_room","bedroom","youth_hostel"]
# bedroom = ["bedchamber","hotel_room","youth_hostel"]
# youth_hostel = ["bedchamber","bedroom","hotel_room"]
# quest_room = ["bedchamber","bedroom","hotel_room","quest_room"]
# restaurant = ["dining_room","dining_hall","banquet_hall","sushi_bar","pizzeria"]
#orchard = ["field","path","garden"]

RelatedWord["castle"] = castle
RelatedWord["bridge"] = bridge
RelatedWord["palace"] = palace
RelatedWord["church"] = church

# RelatedWord["street"] = street
# RelatedWord["crosswalk"] = crosswalk
# RelatedWord["park"] = park
# RelatedWord["lawn"] = lawn
# RelatedWord["building"] = building
# RelatedWord["house"] = house
# RelatedWord["square"] = square
# RelatedWord["hotel_room"] = hotel_room
# RelatedWord["bedchamber"] = bedchamber
# RelatedWord["bedroom"] = bedroom
# RelatedWord["youth_hostel"] = youth_hostel
# RelatedWord["quest room"] = youth_hostel
# RelatedWord["restaurant"] = restaurant
# RelatedWord["orchard"] = orchard

class resPlaces(object):
    def __init__(self, data,percentage):
        self.data = data
        self.percentage = percentage

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []
        self.parent = []


    def add_child(self, obj):
        self.children.append(obj)

    def add_parent(self, obj):
        self.parent = obj

def addType(id, value):
    if id in POITypes:
        print('')
        #print('error')
    else:
        POITypes[id] = value

def addChilds(id, value):
    if id in Childs:
        print('')
        #print('error')
    else:
        Childs[id] = value

def addParent(id, value):
    Parent[id] = value

def GenerateTreePOI():

    f = open('poi_types.csv', encoding='UTF-8')

    data = []
    line_count = 0

    for line in f:

        data_line = line.rstrip().split(';')

        if line_count == 0:
            line_count += 1
        else:
                res = data_line[1]
                res = res.replace("\"\"", "\"")
                res = res[:-1]
                res = res[1:]

                jsonObject = json.loads(res)

                englishTranslation = ""
                for i in jsonObject:
                    if i.get('en'):
                        englishTranslation = i.get('en')

                addType(data_line[0],Node(englishTranslation.lower()))

                addChilds(data_line[0],data_line[2].lower())
                GeneralTypeID_dict[str(englishTranslation).lower()] = data_line[0]
                line_count += 1


    for childId in Childs:
        if(Childs.get(childId) != ""):
            test = json.loads(Childs.get(childId))

            for integ in test:
                POITypes.get(str(childId)).add_child(POITypes.get(str(integ)))
                POITypes.get(str(integ)).add_parent(POITypes.get(str(childId)))
    print("Tree build")
    return

def evaluate(resultPlaces):
    # Threshold waarde (zekerheid dat het model is voor het iets toewijst, 25% = 0.25!
    Thresholperc = 0.3

    for i in range(0, len(resultPlaces)):

        if(resultPlaces[i].percentage > Thresholperc):
            if (resultPlaces[i].data != None):
                return resultPlaces[i]

    perc = resultPlaces[0].percentage
    indexObjectNul = 0

    objectNul = resultPlaces[indexObjectNul].data

    while(perc < Thresholperc):
        if (objectNul != None):
            if(objectNul.data != "poi"):
                perc = resultPlaces[indexObjectNul].percentage
                for i in range(indexObjectNul+1, len(resultPlaces)):
                    if(resultPlaces[i].data != None):
                        object = resultPlaces[i].data
                        while(object.data != "poi"):
                            if(objectNul == object):
                                perc = perc + resultPlaces[i].percentage
                                break
                            else :
                                object = object.parent
                if(perc < Thresholperc):
                    objectNul = objectNul.parent
            else:
                while True:
                    indexObjectNul = indexObjectNul + 1
                    objectNul = resultPlaces[indexObjectNul].data
                    perc = resultPlaces[indexObjectNul].percentage
                    if(indexObjectNul == 4):
                        return -1
                    break
        else:
            while True:
                indexObjectNul = indexObjectNul + 1
                objectNul = resultPlaces[indexObjectNul].data
                perc = resultPlaces[indexObjectNul].percentage
                if (indexObjectNul == 4):
                    return -1
                break
        # else:
        #     return -1
    return resPlaces(objectNul,perc)

resultaatList = []

GenerateTreePOI()
tellerGeenGoedeMatch = 0

for file in onlyfiles:

    print('fileName -> {}', file)

    # load the test image

    # if not os.access(img_name, os.W_OK):
    #     img_url = 'http://places.csail.mit.edu/demo/' + img_name
    #     os.system('wget ' + img_url)

    if(file.find(".jpeg") != -1 or file.find(".jpg") != -1 or file.find(".png") != -1):

        # try:
            img = Image.open(mypath + "\\" + file)
            input_img = V(centre_crop(img).unsqueeze(0))
            checkFormat = 0
            # forward pass
            try:
                logit = model.forward(input_img)
                h_x = F.softmax(logit, 1).data.squeeze()
            except:
                checkFormat = -1
                print('RuntimeError: Given groups=1, expected input[1, 1, 224, 224] to have 3 channels, but got 1 channels instead')


            if(checkFormat == 0):
                probs, idx = h_x.sort(0, True)

                print('{} prediction on {}'.format(arch, file))

                resultPlaces = []
                filterData = []
                result = ""

                filterIndex = 0

                # output the prediction
                for i in range(0, 5):
                    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

                for i in range(0, 5):

                    classes_split = classes[idx[i]].split("/")

                    if(POITypes.get(GeneralTypeID_dict.get(classes_split[0])) != None):
                        filterData.append(resPlaces(classes_split[0], probs[i]))
                        filterIndex = filterIndex + 1

                for i in range(0, filterIndex):

                        result = result + "_" + classes[idx[i]]
                        percentage = float('{:.3f}'.format(filterData[i].percentage))
                        resultPlaces.append(resPlaces(POITypes.get(GeneralTypeID_dict.get(filterData[i].data)),percentage))


                while(len(resultPlaces) < 5):
                    resultPlaces.append(resPlaces(POITypes.get(GeneralTypeID_dict.get("poi")), 0))

                resEvaluate = evaluate(resultPlaces)

                if(resEvaluate == -1):
                    #geen goede match gevonden
                    print('geen goed match gevonden')
                else:
                    #goede match gevonden :)
                    if(resEvaluate.data != None):
                        print('Categorie ' + resEvaluate.data.data)




