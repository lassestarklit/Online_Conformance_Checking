from pm4py.objects.petri import petrinet
from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.visualization.petrinet import factory as pn_vis_factory
from pm4py.objects.petri import semantics
from pm4py.algo.conformance.alignments import factory as align_factory
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.log.exporter.xes import factory as xes_exporter




class ProcessController:

    def __init__(self, path):
        """
            :param path: path to directory storing process model file

        """
        self.net, self.initial_marking, self.final_marking = pnml_importer.import_net(path)

    def load_log(self,log_path):
        self.log = xes_importer.import_log(log_path)

    def get_alignment(self):
        alignments = align_factory.apply_log(self.log, self.net, self.initial_marking, self.final_marking)
        return alignments

    def process_log(self,new_log_path):
        alignments = self.get_alignment()

        for trace_no, trace_alignment in enumerate(alignments):


            state = True
            event_pointer = 0
            for trace in trace_alignment["alignment"]:
                if (trace[0] == trace[1]):

                    self.log[trace_no][event_pointer]["concept:state"] = state
                    self.log[trace_no][event_pointer]["concept:move"] = True
                    event_pointer += 1
                else:
                    if (trace[0] != None and trace[0] != ">>"):
                        state = False
                        self.log[trace_no][event_pointer]["concept:state"] = state
                        self.log[trace_no][event_pointer]["concept:move"] = False


                        event_pointer += 1
                    elif (trace[1] != None and trace[1] != ">>"):

                        state = False



        xes_exporter.export_log(self.log, new_log_path)
        print("Log exported")









