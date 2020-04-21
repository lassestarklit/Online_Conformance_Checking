from Process_Controller import ProcessController
from LSTM_Controller import LSTMController
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.log.exporter.xes import factory as xes_exporter

def fit_model_from_log(log_path, log_name):
    lstm_controller=LSTMController()
    log = xes_importer.import_log(log_path + '/' + log_name)

    lstm_controller.create_model()

    for case_index, case in enumerate(log):

        for event_index, event in enumerate(case):
            lstm_controller.prepare_training_data(event)
            lstm_controller.fit_model()

def process_log(process_model, log_path, log_name, new_log_name):
    process_controller = ProcessController(process_model)


    log = xes_importer.import_log(log_path + '/' + log_name)
    trace_with_dev=0
    for case_index, case in enumerate(log):
        num_of_deviations = 0

        process_controller.reset_process_model()
        for event_index, event in enumerate(case):
            event_activity = event['concept:name']
            fired_legally = process_controller.fire_transition(event_activity)
            if not fired_legally: num_of_deviations += 1 # add to counter if illegal move
            event["concept:legal"] = fired_legally
        if num_of_deviations>0:
            trace_with_dev += 1
            print("\nCase index: %d  case id: %s" % (case_index, case.attributes["concept:name"]))
            print("Number of illegal moves: ", num_of_deviations)
    print(trace_with_dev)

    xes_exporter.export_log(log, log_path + "/" + new_log_name)
    print("Processed log exported")


if __name__ == '__main__':

    path_to_process_model = r'/Users/lassestarklit/Library/Mobile Documents/com~apple~CloudDocs/Computer Science and Engineering/2nd semester/Online Conformance Checking/Data/PLG_Data/process_model.pnml'
    path_to_log = r'/Users/lassestarklit/Library/Mobile Documents/com~apple~CloudDocs/Computer Science and Engineering/2nd semester/Online Conformance Checking/Data/PLG_Data'
    log_filename = 'log.xes'
    new_log_filename = 'log_processed.xes'
    #process_log(path_to_process_model, path_to_log, log_filename, new_log_filename)
    fit_model_from_log(path_to_log,new_log_filename)
