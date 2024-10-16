
import cv2
import os
import numpy as np
import dill

"""
Funzione che contiene un ciclo che poi va a estrarre i frame dal video e a 
convertirli in un array. Serve per creare il datataset per gli esperimenti con Deep Learning. 
"""

path_train = "/content/drive/MyDrive/Ricco/Codice/Sperimentazioni/[NEW]Sperimentazioni/Sperimentazione_2(Soggetto Non Visto)/dataset_splittato/train"
path_test = "/content/drive/MyDrive/Ricco/Codice/Sperimentazioni/[NEW]Sperimentazioni/Sperimentazione_2(Soggetto Non Visto)/dataset_splittato/val"


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
            file = open("/content/drive/MyDrive/Ricco/Codice/Sperimentazioni/[NEW]Sperimentazioni/Sperimentazione_2(Soggetto Non Visto)/data_binaries/x_"
                        + mod + "_" + str(i), "wb")
            dill.dump(x[i * j:], file)
            file.close()
        else:
            print(len(x[h:k]))
            file = open("/content/drive/MyDrive/Ricco/Codice/Sperimentazioni/[NEW]Sperimentazioni/Sperimentazione_2(Soggetto Non Visto)/data_binaries/x_"
                        + mod + "_" + str(i), "wb")
            dill.dump(x[h:k], file)
            file.close()

    # Creazione del file binario per le label.
    file = open("/content/drive/MyDrive/Ricco/Codice/Sperimentazioni/[NEW]Sperimentazioni/Sperimentazione_2(Soggetto Non Visto)/data_binaries/y_" + mod, "wb")
    dill.dump(y, file)
    file.close()

"Funzione che serve per convertire i video in array"
def convert_to_array(mod):
    path = ""

    if mod == "val":
        path = path_test
    elif mod == "train":
        path = path_train
        print(path)

    
    array_videos = []
    for (s, video) in enumerate(os.listdir(path)):
        video_array = []
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
                #Verifica esistenza dell'immagine cv2.imwrite(os.path.join("/content/drive/MyDrive/Ricco" , 'video1.jpeg'), image)
                
                "Resize dell'immagine"
                image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
                
                "Eventuale taglio dell'immagine"
                image = image[10:-10, 0:]
                
                
                "Definizione della shape dell'array"
                #cv2.imwrite(os.path.join("/content/drive/MyDrive/Ricco" , 'video3.jpeg'), image)
                image.shape = (30, 50, 1)
                video_array.append(image)

            video_array = np.array(video_array)

            "Creazione dell'etichetta"
            language = int(video[0]) - 1
            array_videos.append([video_array, language])

    x = []
    y = []

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

"Il valore inserito servirà per scegliere la cartella dove sono localizzati i video"
convert_to_array(modality)

from google.colab import drive
drive.mount('/content/drive')