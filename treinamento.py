import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import GridSearchCV

path = os.getcwd()
print('__file__:    ', __file__)
print (path)
pathFile = os.path.dirname(os.path.abspath(__file__))
print (pathFile)
os.chdir(pathFile)
pathFileFotos = pathFile + '/yalefaces/treinamento'
print (pathFileFotos)

eigenface = cv2.face.EigenFaceRecognizer_create()
# eigenface = cv2.face.EigenFaceRecognizer_create(40, 8000)
fisherface = cv2.face.FisherFaceRecognizer_create()
#fisherface = cv2.face.FisherFaceRecognizer_create(3, 2000)
lbphface = cv2.face.LBPHFaceRecognizer_create()
#lbph = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 50)

def build_models():
    models = []
    models.append(('eigenface', eigenface, {'num_components': [0,20,40], 'threshold': [3000, 5000, 8000]}))
    models.append(('fisherface', fisherface, {'num_components': [0,1,3], 'threshold': [2000, 5000, 8000]}))
    models.append(('lbphface', lbphface, {'radius': [0, 1, 2], 'neighbors': [0,1,2], 'grid_x': [7,8,9], 'grid_y': [7,8,9], 'threshold': [30, 50, 100]}))
    return models

def getImagemComId():
    caminhos = [os.path.join(pathFileFotos, f) for f in os.listdir(pathFileFotos)]
    print (caminhos)
    
    faces = []
    ids = []
    for caminhoImagem in caminhos:
       imagemFace = Image.open(caminhoImagem).convert('L')
       imagemNP = np.array(imagemFace, 'uint8')
       id = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject", ""))
       #id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
       ids.append(id)
       faces.append(imagemNP)

    return np.array(ids), faces

ids, faces = getImagemComId()

print(ids)
print(faces)

models = build_models()

# for name, model, params in models:
#     gs = GridSearchCV(model, param_grid=params, scoring='neg_root_mean_squared_error')
#     gs.fit(faces, ids)
#     gs.best_params_


eigenface.train(faces, ids)
eigenface.write(pathFile + '/classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write(pathFile + '/classificadorFisher.yml')

lbphface.train(faces, ids)
lbphface.write(pathFile + '/classificadorLBPH.yml')

print('Treinamento realizado')


