"""Script che valuta l'unione di due modelli (identificazione+lingua) e produrrà come output un file .csv dove si possono consultare le risposte date dal modello addestrato
per l'identificazione e le risposte date dal modello addestrato per il riconoscimento della lingua e le risposte date modello per l'identificazione integrando la lingua parlata.
I modelli presi in cosinderazione sono: Identificazione 49% di accuratezza e Riconoscimento della lingua parlata 74%"""
import numpy as np
from tensorflow.keras import models
import dill
import csv
import pandas as pd

#Path dei csv
path_csv_test_encoded ="C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/Esperimenti/Sperimentazione_Finale/csv_test_encoded.csv"
path_indici = "C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/Esperimenti/Sperimentazione_Finale/indici_soggetti.csv"
path_analisi_prob_alta = "C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/Esperimenti/Sperimentazione_Finale/analisi_soggetto_a_5(fattabene).csv"
path_out_risultati = "C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/Esperimenti/risultati_identity+lang(conblocco_modello74_a_5).csv"

#Path dei modelli salvati
path_modello_identificazione = "C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/Esperimenti/Sperimentazione_Finale/6_batch"
path_modello_riconoscimento_lingua = "C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/Esperimenti/Sperimentazione(SoggettoVisto)/6_batch"

#Accesso ai dati binari per il modello dell'identificazione
x_test_conv = dill.load(open("C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/Esperimenti/Sperimentazione_Finale/data_binaries/x_test_0", "rb"))
y_test_conv = dill.load(open("C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/Esperimenti/Sperimentazione_Finale/data_binaries/y_test", "rb"))

#Accesso ai dati binari per il modello del riconoscimento lingua
x_test_conv_lang = dill.load(open("C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/Esperimenti/Sperimentazione(SoggettoVisto)/data_binaries/x_test_0", "rb"))
y_test_conv_lang = dill.load(open("C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/Esperimenti/Sperimentazione(SoggettoVisto)/data_binaries/y_test", "rb"))


#Funzione che apre il file indici_soggetti.csv producendo le label dei nomi (con i corrispettivi indici) per il modello dell'identificazione
def load_label(path):
        label_name = []
        label_index = []

        with open(path, 'r') as file:
                        reader = csv.reader(file)
                        back = ''
                        for line in reader:
                                if(line[0] != 'Soggetto'):
                                    
                                    if(back != line[0]):
                                        back = line[0]
                                        label_name.append(line[0])
                                        label_index.append(float(line[1]))
 
        return label_name, label_index       


label_name, label_index = load_label(path_indici)



#Funzione che permette di identificare il soggetto reale dato un elemento i-esimo di y_test_conv (precedentemente codificato tramite One Hot Encoding)
def dammi_soggetto_giusto(sogg):
        a_array_int_id = []
        for a in sogg:
                a_array_int_id.append(float(a))

        a_index = a_array_int_id.index(1.0)
        with open(path_csv_test_encoded, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                        if(row[2] == 'video_name'):
                                m = ""
                        else:
                                b_array_str_id = row[4:]
                                b_array_int_id = []
                                for a in b_array_str_id:
                                        b_array_int_id.append(float(a))
                                #print(array_int_id)
                                b_index = b_array_int_id.index(1.0)
                                if(np.array_equal(b_array_int_id, a_array_int_id)):
                                        return row[3]
        return '' 

#Funzione che in base all'indice massimo dato da np.argmax di una predizione del modello di identificazione mi associa al soggetto che si riferisce
def trovaPersona(index):
    for i in range(0, len(label_index), 1):
        if(label_index[i] == float(index)):
            return label_name[i]
    return ''


#Funzione che unisce i due modelli e predice una risposta con l'integrazione della lingua parlata
def integrazioneLingua(risposta_modello, soggetto_reale, pred_lang, frame_video_identity, pred, test):
        rep = 0
        for (i,p) in enumerate(x_test_conv_lang):
                if(np.array_equal(x_test_conv_lang[i], frame_video_identity) and pred < 0.40):
                        index_max_lang = np.argmax(pred_lang[i])
                        with open(path_analisi_prob_alta,'r') as file:
                                reader = csv.reader(file)
                                for line in reader:
                                        if(line[1] != 'Soggetti '):
                                                if(str(test) == line[4]):
                                                        
                                                        name = line[1]
                                                        if(name != ''): 
                                                                if(name[0] == str(index_max_lang + 1)):
                                                                        return name
        return risposta_modello


#Si ottiene la lingua predetta dal modello 
def prendiLingua(pred_lang, frame_video_identity):
        label = ["italiano", "inglese", "tedesco", "spagnolo", "olandese", "russo", 
                "giapponese", "francese"]
        for (i,p) in enumerate(x_test_conv_lang):
                if(np.array_equal(x_test_conv_lang[i], frame_video_identity)):
                        index_max_lang = np.argmax(pred_lang[i])  
                        return label[index_max_lang]

        return ''

#Si ottiene la lingua corretta
def prendiLinguaCorretta(frame_video_identity, label_lingue):
        for (i,p) in enumerate(x_test_conv_lang):
                if(np.array_equal(x_test_conv_lang[i], frame_video_identity)):
                        return label_lingue[y_test_conv_lang[i]]

#Si prende la lingua corretta di un dato video
def prendiPredizioneLang(frame_video_identity, pred):
       for (i,p) in enumerate(x_test_conv_lang):
                if(np.array_equal(x_test_conv_lang[i], frame_video_identity)):
                        predizione = pred[i]
                        return predizione[np.argmax(predizione)]


def prova(pred_corrente, test, soggetto_corretto):
        prob = pred_corrente
        index = np.argmax(prob)
        row_1 = {"Soggetti": trovaPersona(index),
                  "Predizione": prob[index],
                  "Soggetto reale": soggetto_corretto,
                  "Test": test}
        prob[index] = 0.0

        index = np.argmax(prob)
        row_2 = {"Soggetti": trovaPersona(index),
                  "Predizione": prob[index],
                  "Soggetto reale": soggetto_corretto,
                  "Test": test}

        prob[index] = 0.0

        index = np.argmax(prob)
        row_3 = {"Soggetti": trovaPersona(index),
                  "Predizione": prob[index],
                  "Soggetto reale": soggetto_corretto,
                  "Test": test}

        prob[index] = 0.0

        index = np.argmax(prob)
        row_4 = {"Soggetti": trovaPersona(index),
                  "Predizione": prob[index],
                  "Soggetto reale": soggetto_corretto,
                  "Test": test}

        prob[index] = 0.0

        index = np.argmax(prob)
        row_5 = {"Soggetti": trovaPersona(index),
                  "Predizione": prob[index],
                  "Soggetto reale": soggetto_corretto,
                  "Test": test}
        row_vuota = {"Soggetti": '',
                  "Predizione": '',
                  "Soggetto reale": '',
                  "Test": ''}

        return row_1, row_2, row_3, row_4, row_5, row_vuota
        

model_one = models.load_model(path_modello_identificazione)
model_two = models.load_model(path_modello_riconoscimento_lingua)

model_one.summary()





score = model_one.evaluate(x_test_conv, y_test_conv)
score_language = model_two.evaluate(x_test_conv_lang, y_test_conv_lang)


predictions = model_one.predict(x_test_conv)
predictions_lang = model_two.predict(x_test_conv_lang)


max_predictions = []

for p in predictions: 
  max_predictions.append(max(p))


lingue = ['italiano', 'inglese', 'tedesco', 'spagnolo', 'olandese', 'russo', 'giapponese', 'francese']

count = 1
soggetti_corretti = 0
totali_soggetti = 0
lingue_corrette = 0
soggetti_corretti_integrazione_lingua = 0

df_all = pd.DataFrame()

for (i, p) in enumerate(predictions): 
  index_max = np.argmax(predictions[i])

  print("Test", count,
        "Predizione massima identità", max_predictions[i], 
        "Soggetto dato dal modello: ", trovaPersona(index_max),
        "Soggetto con integrazione lingua: " , integrazioneLingua(trovaPersona(index_max), dammi_soggetto_giusto(y_test_conv[i]), predictions_lang, x_test_conv[i], max_predictions[i], count),
        "Lingua data dal modello: ", prendiLingua(predictions_lang, x_test_conv[i]),
        "Lingua corretta: ", prendiLinguaCorretta(x_test_conv[i], lingue),
        "Predizione massima lingua: ", prendiPredizioneLang(x_test_conv[i], predictions_lang),
        "Soggetto corretto: ", dammi_soggetto_giusto(y_test_conv[i]) + "\n")
  new_row = {"Test": count,
        "Predizione massima identità": max_predictions[i], 
        "Soggetto dato dal modello ": trovaPersona(index_max),
        "Soggetto con integrazione lingua " : integrazioneLingua(trovaPersona(index_max), dammi_soggetto_giusto(y_test_conv[i]), predictions_lang, x_test_conv[i], max_predictions[i], count),
        "Lingua data dal modello ": prendiLingua(predictions_lang, x_test_conv[i]),
        "Lingua corretta": prendiLinguaCorretta(x_test_conv[i], lingue),
        "Predizione massima lingua: ": prendiPredizioneLang(x_test_conv[i], predictions_lang),
        "Soggetto corretto ": dammi_soggetto_giusto(y_test_conv[i])}
  df_all = df_all.append(new_row, ignore_index=True)
  count = count + 1
  totali_soggetti += 1

  #Controllo della risposta data con l'integrazione lingua è uguale al soggetto reale
  if(integrazioneLingua(trovaPersona(index_max), dammi_soggetto_giusto(y_test_conv[i]), predictions_lang, x_test_conv[i], max_predictions[i], index_max) == dammi_soggetto_giusto(y_test_conv[i])):
        soggetti_corretti_integrazione_lingua += 1
  
  #Controllo della risposta data dal modello di identificazione è uguale al soggetto reale
  if (trovaPersona(index_max) == dammi_soggetto_giusto(y_test_conv[i])): 
    soggetti_corretti += 1

  #Controllo della risposta data dal modello del riconoscimento lingua è uguale alla lingua reale parlata dal soggetto  
  if(prendiLinguaCorretta(x_test_conv[i], lingue) == prendiLingua(predictions_lang, x_test_conv[i])):
        lingue_corrette += 1


df_all.to_csv(path_out_risultati)

print("Soggetti corretti: ", soggetti_corretti)
print("Totali soggetti: ", totali_soggetti)
print("Totali lingue corrette: ", lingue_corrette)
print("Soggetti corretti integrazione lingua: ", soggetti_corretti_integrazione_lingua)
print("Test loss: ", score[0])
print("Test accuracy identity: ", score[1])
print("Test accuracy language", score_language[1])
print("Test accuracy identity+language", soggetti_corretti_integrazione_lingua/256)
