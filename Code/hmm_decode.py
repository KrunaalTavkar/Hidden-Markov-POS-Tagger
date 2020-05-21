#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:34:00 2020

@author: krunaaltavkar
"""

import sys
from collections import defaultdict
import math

class HMM_decode():
    
    def __init__(self):
        
        self.start_state = "Q0"
        self.transition_matrix = defaultdict(dict)
        self.emission_matrix = defaultdict(dict)
        self.all_test_words = []
        self.very_large_value = 5000
        self.final_output = []
    
    
    def get_model(self):
        
        with open("hmmmodel.txt", 'r', encoding = 'utf-8') as model_file:
            for sentence in model_file:
                if str(sentence.strip("\n")) != "***** <<<<< >>>>> *****":
                    tokens = sentence.strip("\n").split("-->")
                    head_tag = tokens[0]
                    state_transitions = tokens[1].split(" ")
                    del state_transitions[-1]
                    state_tranistion_dict = {}
                    
                    for state in state_transitions:
                        tag, value = state.split("/\::")
                        state_tranistion_dict[tag] = float(value)
                    
                    self.transition_matrix[head_tag] = state_tranistion_dict
                else:
                    break
                
            
            for sentence in model_file:
                tokens = sentence.strip("\n").split("-->")
                head_tag = tokens[0]        
                emissions = tokens[1].split(" ")
                del emissions[-1]
                emission_dict = {}
                
                for emit in emissions:
                    tag, value = emit.split("/\::")
                    emission_dict[tag] = float(value)
                    
                self.emission_matrix[head_tag] = emission_dict
    
    
    def get_test_data(self, input_path):
        
        testing_file = open(input_path, 'r', encoding = 'utf-8')
        for sentence in testing_file:
            sentence = sentence.strip("\n")
            tokens = sentence.split(" ")
            self.all_test_words.append(tokens)
    
    
    def viteri_decoding(self):
        
        self.unique_tags = list(self.transition_matrix.keys())
        self.unique_tags.remove(self.start_state)
        # print(self.unique_tags)
        
        for sequence in self.all_test_words:
            current_tags = []
            seq_len = len(sequence)
            initial_probability = {}
            initial_backtrace_pointer = {}
            all_probabilities = defaultdict(dict)
            backtrace_pointer = defaultdict(dict)
            final_max_value = float('-inf')
            
            for i in range(0,seq_len):
                initial_backtrace_pointer[i] = ''
            
            for i in range(0,seq_len):
                initial_probability[i] = 0
            
            for tag in self.unique_tags:
                all_probabilities[tag].update(initial_probability)
            
            for tag in self.unique_tags:
                backtrace_pointer[tag].update(initial_backtrace_pointer)
                
            for tag in self.unique_tags:
                if sequence[0] in self.emission_matrix[tag]:
                    transition_value = math.log(float(self.transition_matrix[self.start_state][tag]))
                    emission_value = math.log(float(self.emission_matrix[tag][sequence[0]]))
                    all_probabilities[tag][0] = transition_value + emission_value
                
                else:
                    if sequence[0][0].isdigit() and tag == 'N':
                        transition_value = math.log(float(self.transition_matrix[self.start_state][tag]))
                        all_probabilities[tag][0] = transition_value
                    else:
                        transition_value = math.log(float(self.transition_matrix[self.start_state][tag]))
                        constant_value = self.very_large_value
                        all_probabilities[tag][0] = transition_value - constant_value
            
            for i in range(1,seq_len):
                for tag in self.unique_tags:
                    max_value = float('-inf')
                    for new_tag in self.unique_tags:
                        if sequence[i] in self.emission_matrix[tag]:
                            transition_value = math.log(float(self.transition_matrix[new_tag][tag]))
                            emission_value = math.log(float(self.emission_matrix[tag][sequence[i]]))
                            current_value = all_probabilities[new_tag][i-1]  + transition_value + emission_value
                            
                        else:
                            if sequence[i][0].isdigit() and tag =='N':
                                transition_value = math.log(float(self.transition_matrix[new_tag][tag]))
                                current_value = all_probabilities[new_tag][i-1] + transition_value
                            
                            else:
                                transition_value = math.log(float(self.transition_matrix[new_tag][tag]))
                                constant_value = self.very_large_value
                                current_value = all_probabilities[new_tag][i-1] + transition_value - constant_value
        
                        if current_value > max_value:
                            max_value = current_value 
                            all_probabilities[tag][i] = max_value
                            backtrace_pointer[tag][i] = new_tag
                                
        
            final_values = [all_probabilities[tag][seq_len-1] for tag in self.unique_tags]
        
            for i in range(0,len(final_values)):
                if final_values[i] > final_max_value:
                    final_max_value = final_values[i]
                    index = self.unique_tags[i]    
            current_tags.append(index)
            
            for i in range(seq_len-1,0,-1):
                next_tag = backtrace_pointer[index][i] if i == seq_len-1 else backtrace_pointer[current_tag][i]
                current_tag = next_tag
                current_tags.append(current_tag)
        
            current_tags.reverse()
            self.final_output.append(current_tags)
                    
        
    def generate_output(self):
        
        output_file = open("hmmoutput.txt", "w", encoding="utf-8")
        for i in range(0,len(self.all_test_words)):
            for j in range(0,len(self.all_test_words[i])):
                output_file.write(str(self.all_test_words[i][j]) + "/" + str(self.final_output[i][j]) + " ")
            output_file.write("\n")
        output_file.close()
        
    
if __name__ == "__main__":
    input_path = 'hmm-training-data/it_isdt_dev_raw.txt'
    # input_path = sys.argv[1]
    hmm_model = HMM_decode()
    hmm_model.get_test_data(input_path)
    hmm_model.get_model()
    hmm_model.viteri_decoding()
    hmm_model.generate_output()
    
    