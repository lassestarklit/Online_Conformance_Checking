from random import randint



from Process_Controller import ProcessController
from LSTM_Controller import LSTMController
from pm4py.objects.log.importer.xes import factory as xes_importer
import csv
from datetime import datetime
from plot_data import plot
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
            model.train_model(case)
            training_num+=1

    print("Number of testcases:", len(test_cases))
    print("Number of training cases:", training_num)

    model.evaluate(test_cases)


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
    num_targets=[1,2]
    different_models = [True,False]
    for target in num_targets:
        for model_type in different_models:

            model = LSTMController(csv_name, embedding=model_type, num_target=target)
            model.create_model()
            model.load_training_logs(path_to_log, training_logs)
            model.load_test_logs(path_to_log, testing_logs)
            model.train_model()
            model.evaluate()


if __name__ == '__main__':

    path_to_process_model = r'/Users/lassestarklit/Library/Mobile Documents/com~apple~CloudDocs/Computer Science and Engineering/2nd semester/Online Conformance Checking/Data/PLG_Data/process_model.pnml'
    path_to_log = r'/Users/lassestarklit/Library/Mobile Documents/com~apple~CloudDocs/Computer Science and Engineering/2nd semester/Online Conformance Checking/Data/PLG_Data'

    #Process logs
    '''files=['80','100','150','200','500','750']
    file_names = ['test-' + activity for activity in files]

    for file in file_names:
            process_log(path_to_process_model, path_to_log, file + '.xes', file + '_processed.xes')'''

    # Create CSV
    current_time = datetime.now()
    dt_string = current_time.strftime("%d%m%Y_%H%M%S")
    csv_name = 'Results_' + dt_string + '.csv'
    # Lav csv fil
    with open(csv_name, 'a', newline='') as csvfile:
        fieldnames = ['errors_pr_mil', 'data_processing', 'move/state discriminating', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()


    #run LSTM
    training_logs = ['500','150','100']
    testing_logs = ['80','100','200','750']
    run_lstm(csv_name,path_to_log,training_logs,testing_logs)

    plot(csv_name)


