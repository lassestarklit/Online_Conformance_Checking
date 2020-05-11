# Ignore  the warnings
import warnings

import keras

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences
import csv

from keras import backend as K


class LSTMController:

    def __init__(self, csv_name,err_pr_mil,labels=None,embedding=True,num_target=1):

        if labels is None:
            labels = ['A', 'H', 'J', 'C', 'I', 'K', 'E', 'F', 'G', 'D', 'B']
        self.labels = ['Activity ' + activity for activity in labels]
        self.err_pr_mil = err_pr_mil


        self.X_train = []

        self.X_train_list=[]
        self.y_train_list = []
        self.y_train = np.empty((0,num_target))


        self.embedding = embedding
        self.num_target=num_target

        self.csv_name=csv_name







    def create_model(self):

        # create the model
        # For a single-input model with 2 classes (binary classification):
        '''Embedding Layer: Here we specify the embedding size for our categorical variable. I have used 3 in this case, if we were to increase this it will capture more details on the relationship between the categorical variables. Jeremy Howard suggests the following solution for choosing embedding sizes:
        # m is the no of categories per feature
        embedding_size = min(50, m+1/ 2)
        https://towardsdatascience.com/deep-embeddings-for-categorical-variables-cat2vec-b05c8ab63ac0'''
        embedding_size = 6
        no_features = len(self.labels)
        self.model = Sequential()
        if self.embedding:
            self.model.add(Embedding(input_dim=no_features, output_dim=embedding_size, input_length=1))

        self.model.add(LSTM(100))
        self.model.add(Dense(self.num_target, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def one_hot_encode(self, input_activity):
        """

        :param input_activity: activity label to one-hot-encode
        :return:
        """
        one_hot_encoded = [1 if activity == input_activity else 0 for activity in self.labels]

        return one_hot_encoded

    def reset_training_data(self):
        self.X_train_list=[]
        self.X_train = []
        self.y_train = np.empty((0,self.num_target))



    def train_model(self):
        if self.embedding:

            X_train = pad_sequences(self.X_train)
            print("embedded", (X_train))

        else:
            print("one-hot",self.X_train)
            print("samples of one-hot",len(self.X_train))
            print("timesteps of one-hot",len(self.X_train[0]))
            print(print("features of one-hot",len(self.X_train[0][0])))
            self.X_train = np.reshape(self.X_train, (len(self.X_train), len(self.X_train[0]), len(self.X_train[0][0])))


        #self.fit_model()


    def fit_model(self):

        # Batch size 1 (Stochastic Gradient Descent) since stream data
        self.model.fit(self.X_train, self.y_train,verbose=0)

    def prepare_data_embedding(self, trace):
        """
        :param data: trace from XES log
        :return:
        This function takes a trace as input and defines an event as an array and defines X_train as array of arrays
        """


        X_train_list=[]
        for event in trace:

            activity_label = event["concept:name"]
            state_bool = event["concept:state"]
            move_bool = event["concept:move"]


            #Give value to categorical feature
            for index, activity in enumerate(self.labels):
                if activity == activity_label:

                    X_train_list.append(index)

            #insert row in y train
            target = [0 if state_bool else 1]
            if self.num_target==2:
                target.append(0 if move_bool else 1)
            y_train = np.array(target)

            self.y_train = np.vstack((self.y_train,y_train))


        self.X_train.append(X_train_list)


    def prepare_data_one_hot(self,trace):
        """
        :param data: trace from XES log
        :return:
        This function takes a trace as input and defines an event as an array and defines X_train as array of arrays
        """

        X_train=[]
        for event in trace:

            activity_label = event["concept:name"]
            state_bool = event["concept:state"]
            move_bool = event["concept:move"]

            event_one_hot = self.one_hot_encode(activity_label)
            X_train.append(event_one_hot)

            #insert row in array for X_train



            # insert row in y train
            target = [0 if state_bool else 1]
            if self.num_target == 2:
                target.append(0 if move_bool else 1)
            y_train = np.array(target)

            self.y_train = np.vstack((self.y_train,y_train))
        print(self.X_train.append(X_train))



    def evaluate(self,cases):

        X_test = []
        y_test = np.empty((0, self.num_target))


        for case in cases:
            for event in case:
                activity_label = event["concept:name"]
                state_bool = event["concept:state"]
                move_bool = event["concept:move"]

                if self.embedding:
                    # Give value to categorical feature
                    for index, activity in enumerate(self.labels):
                        if activity == activity_label:
                            X_test.append([index])
                else:
                    X_test_list = self.one_hot_encode(activity_label)
                    X_test.append(X_test_list)



                target = [0 if state_bool else 1]
                if self.num_target == 2:
                    target.append(0 if move_bool else 1)
                y = np.array(target)

                y_test = np.vstack((y_test,y))

        if self.embedding:
            X_test = pad_sequences(X_test)

        if not self.embedding:
            X_test=np.reshape(X_test, (len(X_test),len(X_test[0]),len(X_test[0][0])))
        print(X_test)

        #yhat = self.model.predict_classes(X_test, verbose=0)
        #print(yhat)
        # evaluate the model
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        print("___Embedding___" if self.embedding else "___One hot___")
        print("___Move/State discriminating___" if self.num_target==2 else "____Non discriminating___")
        print('Test loss:', loss)
        print('Test accuracy:', accuracy)
        print('\n')


        move_state_discriminating = True if self.num_target == 2 else False
        categorical_data = "Embedded" if self.embedding else "One hot encoding"
        csv_row=[self.err_pr_mil, categorical_data,move_state_discriminating,accuracy]
        with open(self.csv_name, 'a',newline='\n') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(csv_row)








