import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import codecs
import csv
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from tensorflow.keras import regularizers
import json
import chardet

#Categories
#0-sport, 1-kultur, 2-techtud


def test_with_diff_sources(model):
    path = "C:\\Suli\\Tesztfeladat\\"
    file_name = path + "test_data_from_diff_sources.csv"

    df = pd.read_csv(file_name,encoding='utf8')
    #delete id
    del df['Id']

    #prepare to make categorical values
    df['Topic'] = df['Topic'].replace(['sport'],'0')
    df['Topic'] = df['Topic'].replace(['kultur'],'1')
    df['Topic'] = df['Topic'].replace(['techtud'],'2')
    df = df.astype({"Topic" : int})


    df = shuffle(df)

    #transform to numpy list
    labels =  df["Topic"].to_numpy()
    texts = df["Text"].to_numpy()

    #a,e,i,o,u
    #Filter the text
    texts = first_filter_text(texts)
    texts = second_filter_text(texts)

    #for text in texts:
    #print(text)

    #make categorical values
    labels = tf.keras.utils.to_categorical(labels, num_classes = 3)

    vocab_size = 140000
    tokenizer = Tokenizer(num_words= vocab_size,oov_token="<OOV>")
    #make numbers from words
    tokenizer.fit_on_texts(texts)

    #make sequences (array of integers, the tokenizer parsed an int to each word)
    sequences = tokenizer.texts_to_sequences(texts)

    max_length = 450

    #make same length arrays
    padded = pad_sequences(sequences, padding = 'post', maxlen=max_length)

    result = model.predict(padded)

    print(result)
    print(labels)





def first_filter_text(texts):
    #Replace special characters
    texts = [re.sub(r'\-|\?|!|\.|\,',' ',text) for text in texts]
    #Upper to lowercase
    texts = [text.lower() for text in texts]
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
    #Remove suffixes
    texts = [re.sub('.tt |.al |.el |ban |ben |bol |ba |be |rol |tol |nak |nek |ana |ene |na |ne |ni |va |ve |.bb |ja |je |ra |re ',' ',text) for text in texts]
    return texts
def preprocessing():
    
    path = "C:\\Suli\\Tesztfeladat\\"
    scraped_file_name = 'data.json'
    scraped_file_with_path = path + scraped_file_name
    #Load the json file
    with open(scraped_file_with_path, encoding= 'utf8') as f:
        scraped_data = json.load(f)

    #Read csv to pd df
    df = pd.read_csv(path + "texts.csv",encoding='utf8')

    #remove ID 
    df.rename(columns={"Unnamed: 0": "Id"}, inplace = True)
    del df['Id']

    #Check whether there is null value
    #print(df.info())
    #count group by Topic 
    #print(df['Topic'].value_counts())


    #make pd df from scraped data
    df2 = pd.DataFrame(scraped_data)
    df2.rename(columns={0: "Topic", 1 : "Text" }, inplace = True)

    #Merge df-s
    df = df.append(df2)


    #prepare to make categorical values
    df['Topic'] = df['Topic'].replace(['sport'],'0')
    df['Topic'] = df['Topic'].replace(['kultur'],'1')
    df['Topic'] = df['Topic'].replace(['techtud'],'2')
    df = df.astype({"Topic" : int})

    #shuffle
    df = shuffle(df)

    #transform to numpy list
    labels =  df["Topic"].to_numpy()
    texts = df["Text"].to_numpy()

    #make categorical values
    labels = tf.keras.utils.to_categorical(labels, num_classes = 3)



    #a,e,i,o,u
    #Filter the text
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


    print("Min length: ", minimum)
    print("25 length: ", percent25)
    print("Median length: ", half)
    print("75 length: ", percent75)
    print("Max length: ", maximum)
def compute(texts, labels):

    #Significantly influence the training speed
    vocab_size = 140000

    tokenizer = Tokenizer(num_words= vocab_size,oov_token="<OOV>")
    #make numbers from words
    tokenizer.fit_on_texts(texts)

    #number of different words
    #print("Number of different words: ",len(tokenizer.word_index))

    #make sequences (array of integers, the tokenizer parsed an int to each word)
    sequences = tokenizer.texts_to_sequences(texts)

    max_length = 450

    #make same length arrays
    padded = pad_sequences(sequences, padding = 'post', maxlen=max_length)

    #split data into train-test 70%-30%
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

    #check the size of the model
    #model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #make a callback to prevent overfittting; stop at 99.8% training accuracy
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>0.998):
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()


    num_epochs = 10
    history = model.fit(training_x,training_y,epochs=num_epochs, validation_data=(testing_x, testing_y),callbacks=[callbacks])

    return model, history
def show(history):
    #Show the result of the training
    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_'+string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_'+string])
        plt.show()
    
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")



def main():
  texts, labels = preprocessing()
  #statistics(texts)
  model, history = compute(texts, labels)
  #show(history)
  test_with_diff_sources(model)




if __name__ == "__main__":
  main()
