"""Script per la valutazione del modello addestrato per il riconoscimento della lingua parlata"""
import numpy as np
from tensorflow.keras import models
import sklearn.metrics as sm
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics
import matplotlib.pyplot as plt
import dill

label = ["italiano", "inglese", "tedesco", "spagnolo", "olandese", "russo", 
         "giapponese", "francese"]


corrette_lingua = {"italiano": 0, "inglese": 0, "tedesco": 0, "spagnolo": 0, 
        "olandese": 0, "russo": 0, "giapponese": 0, "francese": 0}
totali_lingua = {"italiano": 0, "inglese": 0, "tedesco": 0, "spagnolo": 0, 
        "olandese": 0, "russo": 0, "giapponese": 0, "francese": 0}

predizioni_per_matrice = []
lingua_reale_per_matrice = []
total = 0
correct = 0
count = 1
risp_corrette = 0
risp_tot = 0

#Caricamento dei dati binari utilizzati nel modello 
x_test_conv = dill.load(open("C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/Esperimenti/Sperimentazione(SoggettoVisto)/data_binaries/x_test_0", "rb"))
y_test_conv = dill.load(open("C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/Esperimenti/Sperimentazione(SoggettoVisto)/data_binaries/y_test", "rb"))

x_train_conv = dill.load(open("C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/Esperimenti/Sperimentazione(SoggettoVisto)/data_binaries/x_train_0", "rb"))
y_train_conv = dill.load(open("C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/Esperimenti/Sperimentazione(SoggettoVisto)/data_binaries/y_train", "rb"))


#Caricamento del modello
model_one = models.load_model("C:/Users/CasaLabDell2/Desktop/Maurizio_Ricco/Esperimenti/Sperimentazione(SoggettoVisto)/6_batch")

model_one.summary()





score = model_one.evaluate(x_test_conv, y_test_conv)

predictions = model_one.predict(x_test_conv)

max_predictions = []



for p in predictions: 
  max_predictions.append(max(p))



for (i, p) in enumerate(predictions): 
  #numero_di_risposte_esatte/numero_di_risposte_totali
  print("Test", count,
        "Predizione massima", max_predictions[i], 
        "Lingua: ", label[np.argmax(predictions[i])], 
        "Lingua corretta: ", label[y_test_conv[i]] + "\n")
  count = count + 1

  lingua_reale_per_matrice.append(label[y_test_conv[i]])
  predizioni_per_matrice.append(label[np.argmax(predictions[i])])

  valore = totali_lingua.get(label[np.argmax(predictions[i])])
  valore += 1
  risp_tot = valore
  totali_lingua[label[np.argmax(predictions[i])]] = valore

  
  if (label[y_test_conv[i]] == label[np.argmax(predictions[i])]): 
    valore1 = corrette_lingua.get(label[y_test_conv[i]])
    valore1 += 1
    risp_corrette += 1
   
    corrette_lingua[label[y_test_conv[i]]] = valore1

#Stampa matrice di confusione
conf_matrix = confusion_matrix(y_true =lingua_reale_per_matrice, y_pred = predizioni_per_matrice)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

print(risp_corrette)


print("Corrette per linguae: ", corrette_lingua)
print("Totali per lingua: ", totali_lingua)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])
