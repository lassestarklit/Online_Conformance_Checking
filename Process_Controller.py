import os

from pm4py.algo.conformance import tokenreplay
from pm4py.objects.petri import petrinet

from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.visualization.petrinet import factory as pn_vis_factory
from pm4py.algo.conformance.alignments import factory as align_factory

from pm4py.objects.petri import semantics
from pm4py.algo.conformance.tokenreplay import factory as token_replay
import pm4py.algo.conformance.tokenreplay


class ProcessController:

    def __init__(self,path):
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


    def load_log(self,path):
        '''

        :param path: path to directory storing event log
        :return:
        '''



    def show_petri_net(self):
        gviz = pn_vis_factory.apply(self.net, self.initial_marking, self.final_marking)
        pn_vis_factory.view(gviz)


    def fire_transition(self,transition_to_fire):
        """

        :param transition: Transition which is desired to fire
        :return: Transition was legally fired: True
                Else: False
        """
        transition_fired = False

        enabled_transitions = semantics.enabled_transitions(self.net, self.marking)

        for transition in enabled_transitions:

            if str(transition) == transition_to_fire:
                #print("Transition {0} fired".format(transition_to_fire))
                self.marking = semantics.execute(transition, self.net, self.marking)
                transition_fired = True

        if not transition_fired:
            #print("Transition {0} fired with weak execute".format(transition_to_fire))
            for transition in self.net.transitions:
                if str(transition) == transition_to_fire:
                    self.marking = semantics.weak_execute(transition, self.marking)

        #Check if newly enabled transition not a "Real transition"
        enabled_transitions = semantics.enabled_transitions(self.net, self.marking)
        for transition in enabled_transitions:

            if self.is_split_join(transition):

                if self.is_split(transition): #If split put token in all places it splits to
                    self.marking = semantics.execute(transition, self.net, self.marking)

                elif self.is_join(transition): # If join check if transition is enabled.
                    if semantics.is_enabled(transition,self.net, self.marking): #If transition is enabled fire.
                        self.marking = semantics.execute(transition, self.net, self.marking)

        return transition_fired


    def is_split_join(self,transition):
        """

        :param transition: transition or split/join
        :return: if split :  True
                 else : False
        """
        #transition label is not defined in split
        if transition.label == None:
            return True
        else:
            return False

    def is_split(self,transition):
        """
                :param transition: transition or split
                :return: if split :  True
                         else : False

        """
        #If transition has more than 1 outgoing arc then it is a split
        if len(transition.out_arcs)>1:
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







