from random import randint



from Process_Controller import ProcessController
from LSTM_Controller import LSTMController
from pm4py.objects.log.importer.xes import factory as xes_importer
import csv
from datetime import datetime
from plot_data import plot


def process_log(process_model, log_path, log_name, new_log_name):
    process_controller = ProcessController(process_model)
    process_controller.load_log(log_path + '/' + log_name)
    process_controller.process_log(log_path + '/' + new_log_name)


def run_lstm(csv_name,path_to_log,logs):
    """

    :param csv_name:
    :param path_to_log:
    :param logs:
    :return:
    """

    feature_process = ['One hot encoding', 'Freq one hot encoding']
    for process in feature_process:
        model = LSTMController(csv_name, feature_process=process, num_target=1)
        model.load_split_logs(path_to_log, logs)
        model.create_model()
        model.train_model()
        model.evaluate()





if __name__ == '__main__':

    # Process data logs
    path_to_process_model = r'process models/process_model.pnml'
    path_to_log = r'logs/'

    '''files=['80','100','150','200','500','750']
    file_names = ['test-' + activity for activity in files]

    for file in file_names:
            process_log(path_to_process_model, path_to_log, file + '.xes', file + '_processed.xes')'''

    # Create CSV
    current_time = datetime.now()
    dt_string = current_time.strftime("%d%m%Y_%H%M%S")

    csv_name = 'results/' + dt_string + '.csv'
    # Give attribute labels
    with open(csv_name, 'a', newline='') as csvfile:
        fieldnames = ['errors_pr_mil', 'feature process', 'move/state discriminating', 'Accuracy','AUC',"F1"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()


    #run LSTM
    #Define which logs the model is trained on. Can add multiple in one list like: ['80','500']
    dif_logs = [['80'],['100'],['150'],['200'],['500'],['750']]
    for logs in dif_logs:
        run_lstm(csv_name,path_to_log,logs)


    performance_metrics_to_plot = ["AUC","F1","Accuracy"]
    for metric in performance_metrics_to_plot:
        plot(csv_name, dif_logs, metric)



