from random import randint



from Process_Controller import ProcessController
from LSTM_Controller import LSTMController
from pm4py.objects.log.importer.xes import factory as xes_importer
import csv
from datetime import datetime

def fit_model_from_log(log_path, log_name,csv_name,err_pr_mil,embedding=True,num_target=1):

    model=LSTMController(csv_name,err_pr_mil,embedding=embedding,num_target=num_target)

    log = xes_importer.import_log(log_path + '/' + log_name)

    model.create_model()

    test_cases=[]
    training_num=0
    #20% are testcases
    testcases=20
    for case_index, case in enumerate(log):
        random=randint(1,100)
        if random<=testcases and len(test_cases)<180:
            test_cases.append(case)
        else:
            if embedding:
                model.prepare_data_embedding(case)
            else:
                model.prepare_data_one_hot(case)
            training_num+=1
    model.train_model()

    print("Number of testcases:", len(test_cases))
    print("Number of training cases:", training_num)

    model.evaluate(test_cases)


def process_log(process_model, log_path, log_name, new_log_name):
    process_controller = ProcessController(process_model)
    log = xes_importer.import_log(log_path + '/' + log_name)
    process_controller.load_log(log_path + '/' + log_name)
    process_controller.process_log(log_path + '/' + new_log_name)




def test():
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    import keras
    # For a single-input model with 10 classes (categorical classification):

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Generate dummy data
    import numpy as np
    data = np.random.random((1000, 100))
    labels = np.random.randint(10, size=(1000, 1))
    print(data)
    print(labels)

    # Convert labels to categorical one-hot encoding
    one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
    print(one_hot_labels)
    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data, one_hot_labels, epochs=10, batch_size=32)


if __name__ == '__main__':

    path_to_process_model = r'/Users/lassestarklit/Library/Mobile Documents/com~apple~CloudDocs/Computer Science and Engineering/2nd semester/Online Conformance Checking/Data/PLG_Data/process_model.pnml'
    path_to_log = r'/Users/lassestarklit/Library/Mobile Documents/com~apple~CloudDocs/Computer Science and Engineering/2nd semester/Online Conformance Checking/Data/PLG_Data'
    files=['80','100','150','200','500','750']

    file_names = ['test-' + activity for activity in files]
    # Create CSV
    current_time = datetime.now()
    dt_string = current_time.strftime("%d%m%Y_%H%M%S")
    csv_name = 'Results_' + dt_string + '.csv'
    # Lav csv fil
    with open(csv_name, 'a', newline='') as csvfile:
        fieldnames = ['errors_pr_mil', 'data_processing', 'move/state discriminating', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
    for index,file in enumerate(file_names):
        #process_log(path_to_process_model, path_to_log, file + '.xes', file + '_processed.xes')



        #Embedding
        fit_model_from_log(path_to_log, file + '_processed.xes',csv_name,err_pr_mil=files[index])
        fit_model_from_log(path_to_log, file + '_processed.xes',csv_name,err_pr_mil=files[index],num_target=2)
        #One-hot
        fit_model_from_log(path_to_log,  file + '_processed.xes',csv_name,err_pr_mil=files[index],embedding=False)
        fit_model_from_log(path_to_log, file + '_processed.xes', csv_name,err_pr_mil=files[index],embedding=False,num_target=2)

