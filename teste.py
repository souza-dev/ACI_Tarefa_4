import cv2
import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
import pandas as pd

# Funcao para busca de arquivos
def find(name, path):
    for root, dirs, files in os.walk(path):
        if (name in files) or (name in dirs):
            print("O diretorio/arquivo {} encontra-se em: {}".format(name, root))
            return os.path.join(root, name)
    # Caso nao encontre, recursao para diretorios anteriores
    return find(name, os.path.dirname(path))

# Importar arquivo XML
cv2path = os.path.dirname(cv2.__file__)
print(cv2path)
#C:\Python37\lib\site-packages\cv2
haar_path = find('haarcascades', cv2path)
print(haar_path)
#C:\opencv\opencv4.5.2\build\etc\haarcascades
xml_name = 'haarcascade_frontalface_alt2.xml'
xml_path = os.path.join(haar_path, xml_name)
print(xml_path)
#C:\opencv\opencv4.5.2\build\etc\haarcascades\haarcascade_frontalface_alt2.xml

detectorFace = cv2.CascadeClassifier(xml_path)

path = os.getcwd()
print('__file__:    ', __file__)
print (path)
pathFile = os.path.dirname(os.path.abspath(__file__))
print (pathFile)
os.chdir(pathFile)
pathFileFotos = pathFile + '/yalefaces/teste'
print (pathFileFotos)

#73%
eigenFace = cv2.face.EigenFaceRecognizer_create()
eigenFace.read(pathFile + '/classificadorEigen.yml')

#96%
fisherFace = cv2.face.FisherFaceRecognizer_create()
fisherFace.read(pathFile + '/classificadorFisher.yml')

#66%
LBPHFace = cv2.face.LBPHFaceRecognizer_create()
LBPHFace.read(pathFile + '/classificadorLBPH.yml')

def getImagemComId():
    caminhos = [os.path.join(pathFileFotos, f) for f in os.listdir(pathFileFotos)]
    faces = []
    ids = []
    names = []
    for caminhoImagem in caminhos:
        name = (caminhoImagem.split("/")[-1])
        imagemFace = Image.open(caminhoImagem).convert('L')
        imagemNP = np.array(imagemFace, 'uint8')
        id = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject", ""))
        names.append(name)
        ids.append(id)
        faces.append(imagemNP)
    return names, np.array(ids), faces

names, ids_test, faces_test = getImagemComId()

df = pd.DataFrame({
    'ID_CORRETO': [], 
    'ID_PRED_EF': [], 
    'CONFIAN_EF': [],
    'ID_PRED_FF': [], 
    'CONFIAN_FF': [],
    'ID_PRED_LBPH': [], 
    'CONFIAN_LBPH': [],
    })

corretosEF = 0
corretosFF = 0
corretosLBPH= 0

for indice in range(len(ids_test)):
    id_EF, confianca_EF = eigenFace.predict(faces_test[indice])
    id_FF, confianca_FF = fisherFace.predict(faces_test[indice])
    id_LBPH, confianca_LBPH = LBPHFace.predict(faces_test[indice])
    df = df.append({
        'ID_CORRETO': names[indice], 
        'ID_PRED_EF': int(id_EF), 
        'CONFIAN_EF': round(confianca_EF, 2),
        'ID_PRED_FF': int(id_FF), 
        'CONFIAN_FF': round(confianca_FF, 2),
        'ID_PRED_LBPH': int(id_LBPH), 
        'CONFIAN_LBPH': round(confianca_LBPH, 2),
        },         
        ignore_index=True)
    if id_EF == ids_test[indice]:
        corretosEF += 1
    if id_FF == ids_test[indice]:
        corretosFF += 1
    if id_LBPH == ids_test[indice]:
        corretosLBPH += 1



print(df)
print(pathFile+'/out.csv')
df.to_csv(pathFile+'/out.csv', index=False)
print('Acurácias')
print('EigenFaces: {:.2f}%'.format(100*corretosEF/len(ids_test)))
print('FisherFaces: {:.2f}%'.format(100*corretosFF/len(ids_test)))
print('LBPHFaces: {:.2f}%'.format(100*corretosLBPH/len(ids_test)))