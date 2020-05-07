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
        self.marking = petrinet.Marking()

    def reset_process_model(self):
        self.marking = petrinet.Marking()
        # Find first place and put marking
        for place in self.net.places:
            if place.in_arcs == set():
                self.marking[place] = 1

    def show_petri_net(self):
        gviz = pn_vis_factory.apply(self.net, self.initial_marking, self.final_marking)
        pn_vis_factory.view(gviz)

    def fire_transition(self, transition_to_fire):
        """

        :param transition: Transition which is desired to fire
        :return: Transition was legally fired: True
                Else: False
        """
        transition_fired = False

        enabled_transitions = semantics.enabled_transitions(self.net, self.marking)

        for transition in enabled_transitions:

            if str(transition) == transition_to_fire:
                self.marking = semantics.execute(transition, self.net, self.marking)
                transition_fired = True

        if not transition_fired:

            for transition in self.net.transitions:
                if str(transition) == transition_to_fire:
                    self.marking = semantics.weak_execute(transition, self.marking)

        # Check if newly enabled transition not a "Real transition"
        enabled_transitions = semantics.enabled_transitions(self.net, self.marking)
        for transition in enabled_transitions:

            if self.is_split_join(transition):

                if self.is_split(transition):  # If split put token in all places it splits to
                    self.marking = semantics.execute(transition, self.net, self.marking)

                elif self.is_join(transition):  # If join check if transition is enabled.
                    if semantics.is_enabled(transition, self.net, self.marking):  # If transition is enabled fire.
                        self.marking = semantics.execute(transition, self.net, self.marking)

        return transition_fired

    def is_split_join(self, transition):
        """

        :param transition: transition or split/join
        :return: if split :  True
                 else : False
        """
        # transition label is not defined in split
        if transition.label is None:
            return True
        else:
            return False

    def is_split(self, transition):
        """
                :param transition: transition or split
                :return: if split :  True
                         else : False

        """
        # If transition has more than 1 outgoing arc then it is a split
        if len(transition.out_arcs) > 1:
            return True
        else:
            return False

    def is_join(self, transition):
        '''
                :param transition: transition or split
                :return: if split :  True
                         else : False

        '''
        # If transition has more than 1 ingoing arc and only 1 outgoing then it is a join
        if len(transition.in_arcs) > 1 and len(transition.out_arcs) == 1:
            return True
        else:
            return False

    def load_log(self,log_path):
        self.log = xes_importer.import_log(log_path)

    def get_alignment(self):
        alignments = align_factory.apply_log(self.log, self.net, self.initial_marking, self.final_marking)
        return alignments

    def process_log(self,new_log_path):
        alignments = self.get_alignment()

        for trace_no, trace_alignment in enumerate(alignments):

            #print("--- new sequence: {0} ---".format(trace_no))
            state = True
            event_pointer = 0
            for trace in trace_alignment["alignment"]:
                if (trace[0] == trace[1]):

                    self.log[trace_no][event_pointer]["concept:state"] = state
                    self.log[trace_no][event_pointer]["concept:move"] = True

                    #print(trace[0] + ", " + ("correct-move, correct-state, {0}".format(event_pointer) if state else "correct-move, faulty-state, {0}".format(event_pointer)))
                    event_pointer += 1
                else:
                    if (trace[0] != None and trace[0] != ">>"):
                        state = False
                        self.log[trace_no][event_pointer]["concept:state"] = state
                        self.log[trace_no][event_pointer]["concept:move"] = False
                        #print(trace[0], "faulty-move: Move on log, faulty-state, {0}".format(event_pointer))

                        event_pointer += 1
                    elif (trace[1] != None and trace[1] != ">>"):
                        #print(trace[1], "faulty-move: Move on model, faulty-state")
                        state = False



        xes_exporter.export_log(self.log, new_log_path)
        print("Log exported")









