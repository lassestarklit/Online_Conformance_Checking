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

    def __init__(self, csv_name,feature_process, labels=None,num_target=1):

        if labels is None:
            labels = ['A', 'H', 'J', 'C', 'I', 'K', 'E', 'F', 'G', 'D', 'B']
        self.labels = ['Activity ' + activity for activity in labels]

        self.training_traces = []
        self.training_targets = []
        self.testing_traces = []
        self.testing_targets = []
        self.validation_traces = []

        self.longest_partial_trace = 0

        self.feature_process = feature_process

        self.num_target=num_target
        self.freq_one_hot= [0 for i in range(len(labels))]



        self.csv_name=csv_name

    def reset_freq_encoding(self):
        self.freq_one_hot = [0 for i in range(len(self.labels))]

    def create_model(self):
        # create the model
        self.model = Sequential()
        self.model.add(LSTM(120,activation='sigmoid'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',metrics.AUC(),metrics.Precision(),metrics.Recall()])

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
        training_perc = 80

        for log_file in logs:
            log=xes_importer.import_log(log_path + '/test-' + log_file + "_processed.xes")

            training_size = len(log) * training_perc / 100
            counter = 0

            for trace in log:
                # find longest trace
                if counter < training_size:
                    list_partial_traces, list_targets = self.split_to_partial(trace)
                    self.training_traces.extend(list_partial_traces)
                    self.training_targets.extend(list_targets)
                else:
                    list_partial_traces, list_targets = self.split_to_partial(trace)
                    self.testing_traces.extend(list_partial_traces)
                    self.testing_targets.extend(list_targets)


                counter += 1







        self.longest_partial_trace = max(len(max(self.training_traces, key=len)),len(max(self.testing_traces, key=len)))

        self.pad_traces()

    def split_to_partial(self,trace):
        list_partial_traces = []
        list_targets = []
        current_trace = []
        for event in trace:
            activity_label = event["concept:name"]
            #Last state will always be false if an illegal move has appeared
            state_bool = event["concept:state"]
            move_bool = event["concept:move"]
            current_trace.append(activity_label)

            #Insert current partial trace in list of partial traces
            list_partial_traces.append(current_trace.copy())
            #insert state in list of targets
            list_targets.append(state_bool)

        return list_partial_traces, list_targets




    def pad_traces(self):
        """Inserts padding traces to match longest trace in log"""


        sets=[self.training_traces,self.testing_traces]
        for data_set in sets:
            for trace in data_set:
                for i in range(self.longest_partial_trace - len(trace)):
                    trace.insert(0,"NA")



    def prepare_data(self,input_data,output_data):
        #transforming dta
        X,y = self.transform_data(input_data,output_data)

        #reshaping data
        X = np.reshape(X, (len(X), self.longest_partial_trace, len(self.labels)))

        y = np.array(y)


        return X,y

    def transform_data(self,input_set,output_set):
        X = []
        y = [1 if target else 0 for target in output_set]
        for partial_trace in input_set:
            partial_encoded=[]
            for event in partial_trace:
                # insert event in trace sequence
                encoded = self.one_hot_encode(event)
                partial_encoded.append(encoded)

            X.append(partial_encoded)



        return X,y

    def train_model(self):

        X,y = self.prepare_data(self.training_traces,self.training_targets)

        self.model.fit(X, y,verbose=0)

    def evaluate(self):
        X_test, y_test = self.prepare_data(self.testing_traces,self.testing_targets)

        # evaluate the model
        loss, accuracy, AUC,precision,recall = self.model.evaluate(X_test, y_test, verbose=0)
        try:
            f1 = 2*precision*recall/(precision+recall)
        except ZeroDivisionError:
            f1=0
        # make prediction
        '''ynew = self.model.predict(X_test)
        # show the inputs and predicted outputs
        for i in range(len(y_test)):
            print(X_test[i])
            print("y_true=%s, y_Predicted=%s" % (y_test[i], ynew[i]))'''


        print(self.feature_process)
        print("___Move/State discriminating___" if self.num_target==2 else "____Non discriminating___")
        print('Test loss:', loss)
        print('Test accuracy:', accuracy)
        print('Test AUC:', AUC)
        print('Test F1:', f1)
        print('\n')


        move_state_discriminating = True if self.num_target == 2 else False

        csv_row=[self.logs,self.feature_process,move_state_discriminating,accuracy,AUC,f1]
        with open(self.csv_name, 'a',newline='\n') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(csv_row)








