import re
import json
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn import svm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import metrics




def test_from_json(tokenizer):
    path = "C:\\Suli\\Tesztfeladat\\"
    scraped_file_name = 'test.json'
    scraped_file_with_path = path + scraped_file_name

    with open(scraped_file_with_path, encoding= 'utf8') as f:
            scraped_data = json.load(f)

    df = pd.DataFrame(scraped_data)
    df.rename(columns={0: "Topic", 1 : "Text" }, inplace = True)


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



    #a,e,i,o,u
    #Filter the text
    texts = first_filter_text(texts)
    texts = second_filter_text(texts)


    #make numbers from words
    tokenizer.fit_on_texts(texts)


    
    #number of different words
    #print("Number of different words: ",len(tokenizer.word_index))

    #make sequences (array of integers, the tokenizer parsed an int to each word)
    sequences = tokenizer.texts_to_sequences(texts)

    max_length = 350


    #make same length arrays
    padded = pad_sequences(sequences, padding = 'post', maxlen=max_length)

    return padded, labels

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
    #labels = tf.keras.utils.to_categorical(labels, num_classes = 3)



    #a,e,i,o,u
    #Filter the text
    texts = first_filter_text(texts)
    texts = second_filter_text(texts)

    return texts, labels


texts, labels = preprocessing()
vocab_size = 140000

tokenizer = Tokenizer(num_words= vocab_size,oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)

max_length = 350    

padded = pad_sequences(sequences, padding = 'post', maxlen=max_length)

training_size = int(len(padded) * 7 / 10)

training_x = padded[:training_size]
training_y = labels[:training_size]

testing_x = padded[training_size:]
testing_y = labels[training_size:]





clf = svm.SVC()
clf.fit(training_x, training_y)

predictions = clf.predict(testing_x)
print("Accuracy:",metrics.accuracy_score(predictions, testing_y))
print("-----------------------")

test_x, test_y = test_from_json(tokenizer)
predictions_train = clf.predict(test_x)
print("Accuracy on data from different source:",metrics.accuracy_score(predictions_train, test_y))




