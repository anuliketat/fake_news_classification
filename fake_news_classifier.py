import os
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
import re
import string
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from textblob import TextBlob as tb # for sentence polarity

# deep learning libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

#  metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.externals import joblib # saving and loading model pkl



class Classifier:
    def __init__(self):
        self.model_name = 'fake_news_classifier'
        self.model_version = 'v1.0.0'

    def __get_data__(self, train_data_path=input("Enter training data location: "), split_size=None):
        try:
            data = pd.read_csv(train_data_path, usecols=[1, 2, 3])
        except Exception as e:
            print(e)
        else:
            if split_size:
                train = data[:int(split_size*(len(data)))]
                val = data[int(split_size*(len(data))):]
               # X = data.drop(['Stance'], axis=1)
               # y = data['Stance']
               # x_train, x_test, y_train, y_test = train_test_split(X, y,
                #                                                    test_size=split_size,
                 #                                                   random_state=9786,
                  #                                                  shuffle=True)

                return train, val, data
            else:
                return data

    def __preprocessing__(self, text):
        """
            decontracting combined words
        """
        # specific
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"can\'t", "can not", text)

        # general
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)

        '''
            Make text lowercase, remove text in square brackets,remove links, remove punctuation
            and remove words containing numbers.
        '''
        #text = re.sub("(\w)([A-Z])", r"\1 \2", text) #inserts space in words like eg:HoaxShah Rukh, Social MediaEminem, FakeEbola
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text) #remove all punctuation
        text = re.sub(r'http\S+', '', text) #remove links
        text = re.sub(r"\([^()]*\)", "", text) # remove words containing brackets
        text = re.sub(r'\w*\d\w*', '', text) #remove words containing numbers
        text = re.sub("[^\x00-\x7F]+", '', text) #removes non english (non ASCII) characters eg: chinese text
        text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text) #removes single letter words
        text = re.sub(r'\W*\b\w{1,2}\b', '', text) #removes 2 letter words
        text = re.sub('man|say|tell|report|claim', '', text)
        text = re.sub('\n', '', text) #removes breaks
        text = re.sub(' +', ' ', text) #removes extra spaces
        text = re.sub(r"^\s+", "", text) #remove spaces from start and ending of text

        # lemmatizing and stopwords removal
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        sent = nlp(text)
        text = ' '.join([token.lemma_ for token in sent if (token.is_stop==False)])

        return text

    def __vectorizer__(self, num_features=10000, nn=False, train=False):
        if not nn:
            tfidf = TfidfVectorizer(max_features=num_features/2, ngram_range=(1, 2))

            return tfidf

        else:
            tokenizer = Tokenizer(num_words=num_features, oov_token='<UNK>')
            return tokenizer

    def __dtree_model__(self, X, y, test_data=None):
        model = DecisionTreeClassifier(random_state=123).fit(X, y)

        dtc = {}
        if test_data is not None:
            predictions = model.predict(test_data)
            dtc['preds'] = predictions

        dtc['model'] = model
        return dtc

    def __conv_model__(self, X, y, test_data=None, x_val, y_val, input_length=500, epochs=20, bs=512):

        model = keras.Sequential()
        model.add(keras.layers.Embedding(10000, 64, input_length=input_length))
        model.add(keras.layers.Conv1D(128, 5, activation="relu"))
        model.add(keras.layers.MaxPooling1D(2))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Conv1D(64, 5, activation="relu"))
        model.add(keras.layers.MaxPooling1D(2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(4, activation='softmax'))

        checkpoint_path = "training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=2)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X, y, epochs=epochs, batch_size=bs, validation_data=(x_val, y_val), verbose=1, callbacks=[cp_callback])

        history_dict = history.history

        keras_model = {}
        if test_data is not None:
            predictions = model.predict_classes(test_data)
            keras_model['preds'] = predictions

        keras_model['summary'] = model.summary()
        keras_model['model'] = history
        keras_model['train_acc'] = history_dict['acc']
        keras_model['val_acc'] = history_dict['val_acc']

        return keras_model

    def Model(self, model='dt'):
        train, val, data = self.__get_data__(split_size=0.2)
        print('cleaning data...')
        for df in [train, val, data]:
            df['Headline'] = df['Headline'].apply(lambda x: self.__preprocessing__(x))
            df['Body'] = df['Body'].apply(lambda x: self.__preprocessing__(x))

        if model=='dt':
            print('Vectorizing the data...')
            head_tfidf = self.__vectorizer__()
            train_head_vec = head_tfidf.fit_transform(train['Headline']).toarray()
            body_tfidf = self.__vectorizer__()
            train_body_vec = body_tfidf.fit_transform(train['Body']).toarray()
            test_head_vec = head_tfidf.transform(val['Headline']).toarray()
            test_body_vec = body_tfidf.transform(val['Body']).toarray()

            # for new data prediction
            data_head_tfidf = self.__vectorizer__()
            data_head_vec = data_head_tfidf.fit_transform(data['Headline']).toarray()
            data_body_tfidf = self.__vectorizer__()
            data_body_vec = data_body_tfidf.fit_transform(data['Body']).toarray()
            data_train = np.hstack((data_head_vec, data_body_vec))

            print('Extracting features...')
            X_train = np.hstack((train_head_vec, train_body_vec))
            x_test = np.hstack((test_head_vec, test_body_vec))
            y_train = train['Stance']
            y_test = val['Stance']

            print('Training Decision Tree...')
            m = self.__dtree_model__(X_train, y_train, x_test)
            mod = self.__dtree_model__(data_train, data['Stance'])

            joblib.dump(mod['model'], "model.pkl")
            print('model saved!')

            #cls_rep = classification_report(y_test, dt_predictions)
            #rep = pd.DataFrame(cls_rep).transpose()

            metrics = {}
            metrics['accuracy'] = accuracy_score(y_test, dt_predictions)
            metrics['f1-score'] = f1_score(y_test, dt_predictions, average='weighted')
            #metrics['report'] = rep

        if model=='nn':
            for df in [train, val, data]:
                df['news'] = df['Headline']+df['Body']
                df = df.drop(['Headline', 'Body'], axis=1)

            x_train = train['news']
            y_train = train['Stance']
            x_test = val['news']
            y_test = val['Stance']

            le = LabelEncoder()
            for data in [y_train, y_test]:
                data = le.fit_transform(data)

            tokenizer = self.__vectorizer__(nn=True)
            tokenizer.fit_on_texts(x_train)
            word_index = tokenizer.word_index

            pad_type = 'post'
            trunc_type = 'post'
            maxlen = 500

            train_sequences = tokenizer.texts_to_sequences(x_train)
            train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_padded = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
            # for new data prediction
            data_tokenizer = self.__vectorizer__(nn=True)
            data_tokenizer.fit_on_texts(data['train'])
            data_sequences = tokenizer.texts_to_sequences(x_train)
            data_train = pad_sequences(data_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
            data_val = data_train[int(0.9*len(data_train)):]
            data_y_val = np.array(data['Stance'])[0.9*len(data_train)):]

            word_index = {k:(v+3) for k, v in word_index.items()}
            word_index['<PAD>'] = 0
            word_index['<START>'] = 1
            #reverse_word_index = {v: k for k, v in word_index.items()}

            X_train = train_padded[:37000]
            x_val = train_padded[37000:]
            y_train = np.array(y_train[:37000])
            y_val = np.array(y_train[37000:])
            x_test = test_padded


            print('Training neural network ...')
            m = self.__conv_model__(X_train, y_train, test_data=x_test, x_val, y_val)
            mod = self.__conv_model__(data_train, np.array(data['Stance']), x_val=data_val, y_val=data_yval)
            keras_pred = model['preds']

            #cls_rep = classification_report(y_test, keras_pred, output_dict=True)
            #rep = pd.DataFrame(cls_rep).transpose()

            metrics = {}
            metrics['accuracy'] = accuracy_score(y_test, keras_pred)
            metrics['f1-score'] = f1_score(y_test, keras_pred, average='weighted')

        _model = {}
        _model['eval'] = m
        _model['train'] = mod
        _model['data'] = data

        print('Training Done!')
        print('\n', metrics)
        return _model, metrics,

    def Predict(self, test_data_path=input("Enter test data location: "), model="dt"):
        data = pd.read_csv(test_data_path, usecols=[1, 2, 3])
        data['Headline'] = data['Headline'].apply(lambda x: self.__preprocessing__(x))
        data['Body'] = data['Body'].apply(lambda x: self.__preprocessing__(x))
        tfidf = self.__vectorizer__()
        data_head_vec = tfidf.fit_transform(data['Headline']).toarray()
        data_body_vec = tfidf.fit_transform(data['Body']).toarray()

        data_vec = np.hstack((data_head_vec, data_body_vec))

        clf = joblib.load("model.pkl")
        predictions = clf.predict(data_vec)
        f1 = f1_score(data['Stance'], predictions, average='weighted')

        print("F1-score: ", f1)

        return predictions
        #model, metrics = self.Model(train_data_path)
        #test_data =
        #model.predict()

cls = Classifier()
_model_data, metrics, = cls.Model()
cls.Predict()















































