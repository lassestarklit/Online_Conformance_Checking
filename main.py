from random import randint



from Process_Controller import ProcessController
from LSTM_Controller import LSTMController
from pm4py.objects.log.importer.xes import factory as xes_importer
import csv
from datetime import datetime
from plot_data import plot


def process_log(process_model, log_path, log_name, new_log_name):
    process_controller = ProcessController(process_model)
    log = xes_importer.import_log(log_path + '/' + log_name)
    process_controller.load_log(log_path + '/' + log_name)
    process_controller.process_log(log_path + '/' + new_log_name)


def run_lstm(csv_name,path_to_log,training_logs,testing_logs):
    """

    :param training_logs: list of names of logs
    :param testing_logs: list of names of logs
    :return:
    """
    model = LSTMController(csv_name, feature_process='One hot encoding', num_target=1)

    model.load_split_logs(path_to_log, training_logs)

    model.create_model()
    model.train_model()
    model.evaluate()

    #Create a model for each fault type (1=faulty move and state combined, 2 is discriminating those)
    '''num_targets=[1,2]
    feature_process = ["Embedding","One hot encoding", "Freq one hot encoding"]
    for target in num_targets:
        for model_type in feature_process:

            model = LSTMController(csv_name, feature_process=model_type, num_target=target)
            model.create_model()
            model.load_training_logs(path_to_log, training_logs)
            #model.load_test_logs(path_to_log, testing_logs)
            #model.train_model()
            #model.evaluate()'''


if __name__ == '__main__':

    path_to_process_model = r'process models/process_model.pnml'
    path_to_log = r'logs/'

    #Process data logs
    '''files=['80','100','150','200','500','750']
    file_names = ['test-' + activity for activity in files]

    for file in file_names:
            process_log(path_to_process_model, path_to_log, file + '.xes', file + '_processed.xes')'''

    # Create CSV
    current_time = datetime.now()
    dt_string = current_time.strftime("%d%m%Y_%H%M%S")

    csv_name = 'results/Results_' + dt_string + '.csv'
    # Lav csv fil
    with open(csv_name, 'a', newline='') as csvfile:
        fieldnames = ['errors_pr_mil', 'data_processing', 'move/state discriminating', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()


    #run LSTM
    #training_logs = ['500','150']
    #testing_logs = ['80','100','200','750']
    training_logs = ['500']
    testing_logs = ['80','100']
    run_lstm(csv_name,path_to_log,training_logs,testing_logs)

    plot(csv_name,training_logs,testing_logs)


