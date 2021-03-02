import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import codecs
import csv
import os
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from tensorflow.keras import regularizers
#0-sport, 1-kultur, 2-techtud


#TODO
# bag-of-words
#SVM

def first_filter_text(texts):

    texts = [re.sub(r'\-|\?|!|\.|\,',' ',text) for text in texts]
    texts = [text.replace('á','a') for text in texts]
    texts = [text.replace('é','e') for text in texts]
    texts = [text.replace('ű','u') for text in texts]
    texts = [text.replace('ü','u') for text in texts]
    texts = [text.replace('ú','u') for text in texts]
    texts = [text.replace('ő','o') for text in texts]
    texts = [text.replace('ó','o') for text in texts]
    texts = [text.replace('ö','o') for text in texts]
    texts = [text.replace('í','i') for text in texts]
    return texts

def second_filter_text(texts):
    texts = [re.sub('.tt |.al |.el |ban |ben |bol |ba |be |rol |tol |nak |nek |ana |ene |na |ne |ni |va |ve |.bb |ja |je |ra |re ',' ',text) for text in texts]
    return texts



def preprocessing():

  path = "C:\\Suli\\Tesztfeladat\\"

  #Beolvasom pandas segitsegevel, hogy leellenorizzem, van-e ures cella
  df = pd.read_csv(path + "texts.csv",encoding='utf8')
  df.rename(columns={"Unnamed: 0": "Id"}, inplace = True)
  del df['Id']
  #megnezni, van-e null
  #print(df.info())

  #témákhoz megnezni, mennyi elem tartozik
  #print(df['Topic'].value_counts())


  #kategoriakat szamokka alakitani
  df['Topic'] = df['Topic'].astype("category")

  df['Topic'] = df['Topic'].replace(['sport'],'0')
  df['Topic'] = df['Topic'].replace(['kultur'],'1')
  df['Topic'] = df['Topic'].replace(['techtud'],'2')
  df = df.astype({"Topic" : int})

  #megkeverem
  df = shuffle(df)

  #visszaalakitom python list-re 
  labels =  df["Topic"].to_numpy()
  labels = tf.keras.utils.to_categorical(labels, num_classes = 3)

  texts = df["Text"].to_numpy()

  #a,e,i,o,u

  texts = first_filter_text(texts)
  texts = second_filter_text(texts)

  return texts, labels




def statistics(texts):
  magic_obj =[]
  for text in texts:
        

    words = text.split()
    x = len(words)
    magic_obj.append((text, x))


  def mySort(e):
    return e[1]

  magic_obj.sort(key = mySort)

  size = (len(magic_obj)-1) 

  minimum = magic_obj[0][1]
  percent25 = magic_obj[int(size/4)][1]
  half = magic_obj[int(size / 2)][1]
  percent75 = magic_obj[int(size / 4 * 3)][1]
  maximum = magic_obj[size][1]


  print("Minimális hossz: ", minimum)
  print("25 hossz: ", percent25)
  print("Medián hossz: ", half)
  print("75 hossz: ", percent75)
  print("Maximum hossz: ", maximum)
  


def compute(texts, labels):


  vocab_size = 65000


  tokenizer = Tokenizer(num_words= vocab_size,oov_token="<OOV>")
  tokenizer.fit_on_texts(texts)

  sequences = tokenizer.texts_to_sequences(texts)

  max_length = 450    
  padded = pad_sequences(sequences, padding = 'post', maxlen=max_length)

  training_size = int(len(padded) * 7 / 10)

  training_x = padded[:training_size]
  training_y = labels[:training_size]

  testing_x = padded[training_size:]
  testing_y = labels[training_size:]

  embedding_dim = 50


  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size,embedding_dim, input_length= max_length),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Conv1D(16, 5, activation='relu'),
      tf.keras.layers.GlobalAveragePooling1D(),
      tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
      tf.keras.layers.Dense(3, activation='softmax')
  ])


  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  #model.summary()



  class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
          if(logs.get('accuracy')>0.998):
              print("\nReached 99.8% accuracy so cancelling training!")
              self.model.stop_training = True

  callbacks = myCallback()


  num_epochs = 15
  history = model.fit(training_x,training_y,epochs=num_epochs, validation_data=(testing_x, testing_y),callbacks=[callbacks])
  return history




def show(history):
  def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
    

  plot_graphs(history, "accuracy")
  plot_graphs(history, "loss")



def main():
  texts, labels = preprocessing()
  statistics(texts)
  history = compute(texts, labels)
  show(history)




if __name__ == "__main__":
  main()



