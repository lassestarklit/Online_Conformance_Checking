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
from pm4py.objects.log.importer.xes import factory as xes_importer
import csv

from keras import backend as K


class LSTMController:

    def __init__(self, csv_name,labels=None,embedding=True,num_target=1):

        if labels is None:
            labels = ['A', 'H', 'J', 'C', 'I', 'K', 'E', 'F', 'G', 'D', 'B']
        self.labels = ['Activity ' + activity for activity in labels]


        self.training_logs = {}
        self.testing_logs = {}

        self.X_train = np.empty((0,len(labels)))
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
        self.X_train = np.empty((0,len(self.labels)))
        self.y_train = np.empty((0,self.num_target))

    def load_training_logs(self,log_path,logs):

        for log_name in logs:
            self.training_logs[log_name]=xes_importer.import_log(log_path + '/test-' + log_name + "_processed.xes")

    def load_test_logs(self,log_path,logs):
        for log_name in logs:
            self.testing_logs[log_name] = xes_importer.import_log(log_path + '/test-' + log_name + "_processed.xes")

    def train_model_one_hot(self):
        for name_of_log, log in self.training_logs.items():
            for trace_index, trace in enumerate(log):
                self.reset_training_data()

                for event in trace:

                    activity_label = event["concept:name"]
                    state_bool = event["concept:state"]
                    move_bool = event["concept:move"]

                    X_train_list = self.one_hot_encode(activity_label)
                    new_X_train = np.array(X_train_list)

                    # insert row in array for X_train

                    self.X_train = np.vstack((self.X_train, new_X_train))

                    # insert row in y train
                    target = [0 if state_bool else 1]
                    if self.num_target == 2:
                        target.append(0 if move_bool else 1)
                    y_train = np.array(target)

                    self.y_train = np.vstack((self.y_train, y_train))

                self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))

                self.fit_model()

        '''
        Tried to make bagdes wh: 
        [[[one-hot-activity-1_1],[one-hot-activity-2_1]...[one-hot-activity-n_1]]
        [[one-hot-activity-1_2],[one-hot-activity-2_2]...[one-hot-activity-n_2]]
        ...
        [[one-hot-activity-1_n],[one-hot-activity-2_n]...[one-hot-activity-n_n]]]
        training_data = []
        for name_of_log,log in self.training_logs.items():
            self.reset_training_data()
            num = 0
            for trace_index, trace in enumerate(log):
                num+=1
                one_badge = []
                for event in trace:
                
                    activity_label = event["concept:name"]
                    state_bool = event["concept:state"]
                    move_bool = event["concept:move"]

                    one_hot_encoded_activity = self.one_hot_encode(activity_label)
                    one_badge.append(one_hot_encoded_activity)

                print(num)
                badge_array = np.array(one_badge)
                self.X_train = np.insert(self.X_train,0, badge_array,1)

            print(self.X_train.shape)'''

    def train_model_embedding(self):
        """
        :param data: trace from XES log
        :return:
        This function takes a trace as input and defines an event as an array and defines X_train as array of arrays
        """
        for name_of_log, log in self.training_logs.items():
            for trace_index, trace in enumerate(log):
                self.reset_training_data()
                X_train_list = []
                for event in trace:

                    activity_label = event["concept:name"]
                    state_bool = event["concept:state"]
                    move_bool = event["concept:move"]

                    # Give value to categorical feature
                    for index, activity in enumerate(self.labels):
                        if activity == activity_label:
                            X_train_list.append(index)

                    # insert row in y train
                    target = [0 if state_bool else 1]
                    if self.num_target == 2:
                        target.append(0 if move_bool else 1)
                    y_train = np.array(target)

                    self.y_train = np.vstack((self.y_train, y_train))

                self.X_train = np.array(X_train_list)
                self.fit_model()

    def train_model(self):

        if self.embedding:
            self.train_model_embedding()
        else:
            self.train_model_one_hot()

    def fit_model(self):

        # Batch size 1 (Stochastic Gradient Descent) since stream data

        self.model.fit(self.X_train, self.y_train,verbose=0)






    def evaluate(self):
        for name_of_log, log in self.testing_logs.items():
            print(name_of_log)
            X_test = []
            y_test = np.empty((0, self.num_target))


            for case in log:
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

            X_test = pad_sequences(X_test)

            if not self.embedding:
                X_test=np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

            #yhat = self.model.predict_classes(X_test, verbose=0)
            #print(yhat)
            # evaluate the model
            loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)

            # make a prediction
            '''ynew = self.model.predict(X_test)
            # show the inputs and predicted outputs
            for i in range(len(y_test)):
                print("y_true=%s, y_Predicted=%s" % (y_test[i], ynew[i]))'''

            print("___Embedding___" if self.embedding else "___One hot___")
            print("___Move/State discriminating___" if self.num_target==2 else "____Non discriminating___")
            print('Test loss:', loss)
            print('Test accuracy:', accuracy)
            print('\n')


            move_state_discriminating = True if self.num_target == 2 else False
            categorical_data = "Embedded" if self.embedding else "One hot encoding"
            csv_row=[name_of_log, categorical_data,move_state_discriminating,accuracy]
            with open(self.csv_name, 'a',newline='\n') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(csv_row)








