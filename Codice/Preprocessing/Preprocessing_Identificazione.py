"""Conversione video in dati binari per la sperimentazione 'Identificazione del soggetto' dando i output .csv delle etichette"""

import cv2
import os
import numpy as np
import dill
import pandas as pd
import csv
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

"""
Funzione che contiene un ciclo che poi va a estrarre i frame dal video e a 
convertirli in un array. Serve per creare il datataset per gli esperimenti con Deep Learning. 
"""

path_train = "/content/drive/MyDrive/Ricco/Codice/Sperimentazioni/[NEW]Sperimentazioni/Sperimentazione_Finale(Identificazione)/dataset_splittato_v2/train"
path_test = "/content/drive/MyDrive/Ricco/Codice/Sperimentazioni/[NEW]Sperimentazioni/Sperimentazione_Finale(Identificazione)/dataset_splittato_v2/test"
path_csv_train="/content/drive/MyDrive/Ricco/csv_train_encoded.csv"
path_csv_test ="/content/drive/MyDrive/Ricco/csv_test_encoded.csv"
path_csv  ="/content/drive/MyDrive/Ricco/csv_train_encoded.csv"
nomi_video = []
etichette = []

def create_binaries(x, y, mod):
    split = input("In quante sezioni dividere i dati ? ")
    print(len(x))

    for i in range(0, int(split)):

        # j indica il numero di file.
        j = int(len(x) / int(split))

        h = i * j
        k = j * (i + 1)

        if i == (int(split) - 1):
            print(len(x[i * j:]))
            file = open("/content/drive/MyDrive/Ricco/x_"
                        + mod + "_" + str(i), "wb")
            dill.dump(x[i * j:], file)
            file.close()
        else:
            print(len(x[h:k]))
            file = open("/content/drive/MyDrive/Ricco/x_"
                        + mod + "_" + str(i), "wb")
            dill.dump(x[h:k], file)
            file.close()

    # Creazione del file binario per le label.
    file = open("/content/drive/MyDrive/Ricco/y_" + mod, "wb")
    dill.dump(y, file)
    file.close()

def assignName(name):
    nomi_video.append(name)
    name_parts = name.split("_")
    identity_name = int(name_parts[0] + name_parts[1] + name_parts[2] + name_parts[3])
    etichette.append(int(identity_name))
    print(identity_name)
    return identity_name

"Funzione che serve per convertire i video in array"
def convert_to_array(mod):
    path = ""

    if mod == "test":
        path = path_test
    elif mod == "train":
        path = path_train

    array_videos = []

    for (s, video) in enumerate(os.listdir(path)):
        print(video)
        video_array = []
        videoName = video
        n_frame = 0

        if video != ".DS_Store":
            cap = cv2.VideoCapture(path +"/"+ video)
            while cap.isOpened():

                ret, image = cap.read()
                n_frame += 1

                if not ret or n_frame > 150:
                    break

                "Parte in cui sono eseguite le operazioni di trasformazione in bianco e nero"
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                "Resize dell'immagine"
                image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
                "Eventuale taglio dell'immagine"
                image = image[10:-10, 0:]
                "Definizione della shape dell'array"
                image.shape = (30, 50, 1)
                video_array.append(image)

            video_array = np.array(video_array)

            "Creazione dell'etichetta"
            identity = []
            identity_intero = []
            with open("/content/drive/MyDrive/encoded_"+mod+".csv") as f:
              reader = csv.reader(f)
              for row in reader:
                if(row[2] == "video_name"): 
                  print("Skippo")
                if(row[2] == videoName):
                  identity = row[4:]
                  print(identity[:10])
                  for a in identity:
                    identity_intero.append(float(a))

            print("Inserisco: ", identity_intero)
            print(len(identity_intero))
            array_videos.append([video_array, identity_intero])

    x = []
    y = []
    z = []
    
    print("Inizio ciclo per divisione degli item e delle etichette")
    for feature, label in array_videos:
        x.append(feature)
        y.append(label)

    x = np.array(x)
    y = np.array(y)

    print("Create binaries")
    create_binaries(x, y, mod)


"Alla domanda rispondere con train o test"
modality = input("Dati per training o per test ? ")
path = ""

if modality == "test":
    path = path_test
elif modality == "train":
    path = path_train

#columns = ["id","video_name", "identity"]
df_all = pd.DataFrame()
for video in os.listdir(path):
    identity_name_parts = video.split(".")[0].split("_")
    id = identity_name_parts[0]
    identity_name = identity_name_parts[1]
    new_rows = {'id': id,
                'video_name': video,
                'identity': identity_name}
    df_all = df_all.append(new_rows, ignore_index=True)

df_all.to_csv("/content/drive/MyDrive/"+modality+".csv",index=False)

df = pd.read_csv("/content/drive/MyDrive/"+modality+".csv")
print(df['identity'])
print(df.info())
one_hot_encoder = OneHotEncoder()
enc_data=pd.DataFrame(one_hot_encoder.fit_transform(df[['identity']]).toarray())
print("____________")
print(one_hot_encoder.fit_transform(df[['identity']]).toarray())
print("____________")
New_df = df.join(enc_data)
print(New_df)
New_df.to_csv("/content/drive/MyDrive/encoded_"+modality+".csv", encoding='utf-8')

path=""
"Il valore inserito servirà per scegliere la cartella dove sono localizzati i video"
convert_to_array(modality)

