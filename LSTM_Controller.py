# Ignore  the warnings
import warnings




warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding,LSTM
from keras import metrics
from keras.preprocessing.sequence import pad_sequences
from pm4py.objects.log.importer.xes import factory as xes_importer
import csv



class LSTMController:

    def __init__(self, csv_name,feature_process, loss_function, labels=None,num_target=1):

        if labels is None:
            labels = ['A', 'H', 'J', 'C', 'I', 'K', 'E', 'F', 'G', 'D', 'B']
        self.labels = ['Activity ' + activity for activity in labels]

        self.training_traces = []
        self.testing_traces = []
        self.validation_traces = []

        self.longest_trace = 0

        self.feature_process = feature_process

        self.num_target=num_target
        self.freq_one_hot= [0 for i in range(len(labels))]


        self.loss_function = loss_function
        self.csv_name=csv_name

    def reset_freq_encoding(self):
        self.freq_one_hot = [0 for i in range(len(self.labels))]

    def create_model(self):

        # create the model

        self.model = Sequential()
        self.model.add(LSTM(100))

        self.model.add(Dense(self.longest_trace, activation='sigmoid'))
        self.model.compile(loss=self.loss_function, optimizer='adam', metrics=['accuracy',metrics.AUC()])




    def one_hot_encode(self, input_activity):
        """
        :param input_activity: activity label to one-hot-encode
        :return:
        """
        one_hot_encoded = [1 if activity == input_activity else 0 for activity in self.labels]

        if self.feature_process == 'Freq one hot encoding':
            for index, freq in enumerate(self.freq_one_hot):
                one_hot_encoded[index] += freq
            self.freq_one_hot = one_hot_encoded
        return one_hot_encoded

    def load_split_logs(self,log_path,logs):
        self.logs = logs
        training_perc=80

        for log_file in logs:
            log=xes_importer.import_log(log_path + '/test-' + log_file + "_processed.xes")

            training_size = len(log) * training_perc / 100
            counter = 0

            for trace in log:
                #find longest trace
                if len(trace) > self.longest_trace:
                    self.longest_trace = len(trace)
                if counter < training_size:
                    self.training_traces.append(trace)
                else:
                    self.testing_traces.append(trace)

                counter += 1

        self.pad_traces()

    def pad_traces(self):
        """Inserts padding traces to match longest trace in log"""
        pad_str = {'concept:name': 'NA', 'concept:state': True, 'concept:move': True}

        sets=[self.training_traces,self.testing_traces,self.validation_traces]
        for data_set in sets:
            for trace in data_set:
                for i in range(self.longest_trace-len(trace)):
                    trace.insert(0,pad_str)


    def prepare_data(self,data_set):
        #transforming dta
        X,y = self.transform_data(data_set)
        #reshaping data
        X = np.reshape(X, (len(data_set), len(data_set[0]), len(self.labels)))
        y = pad_sequences(y)

        return X,y

    def transform_data(self,data_set):
        X = []
        y = []
        for trace in data_set:
            self.reset_freq_encoding()
            trace_activity = []
            trace_target = []
            for event in trace:
                activity_label = event["concept:name"]
                state_bool = event["concept:state"]
                move_bool = event["concept:move"]

                # insert event in trace sequence
                encoded = self.one_hot_encode(activity_label)
                trace_activity.append(encoded)

                # insert target in trace sequence

                target = 1 if state_bool else 0
                if self.num_target == 2:
                    target = [target, 1 if move_bool else 0]

                trace_target.append(target)

            X.append(trace_activity)
            y.append(trace_target)

        return X,y

    def train_model(self):

        X,y = self.prepare_data(self.training_traces)

        self.model.fit(X, y,verbose=0)

    def evaluate(self):
        X_test, y_test = self.prepare_data(self.testing_traces)

        # evaluate the model
        loss, accuracy,AUC = self.model.evaluate(X_test, y_test, verbose=0)


        ''''# make prediction
        ynew = self.model.predict(X_test)
        # show the inputs and predicted outputs
        for i in range(len(y_test)):
            print("y_true=%s, y_Predicted=%s" % (y_test[i], ynew[i]))'''

        print(self.feature_process)
        print("___Move/State discriminating___" if self.num_target==2 else "____Non discriminating___")
        print('Test loss:', loss)
        print('Test accuracy:', accuracy)
        print('Test AUC:', AUC)
        print('\n')


        move_state_discriminating = True if self.num_target == 2 else False

        csv_row=[self.logs, self.loss_function,move_state_discriminating,accuracy,AUC]
        with open(self.csv_name, 'a',newline='\n') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(csv_row)








