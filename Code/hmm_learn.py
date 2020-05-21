#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:56:13 2020

@author: krunaaltavkar
"""

import sys
from collections import Counter
from collections import defaultdict

class HMM_learn():
    
    def __init__(self):
        
        self.all_state_transitions = defaultdict(int)
        self.all_tag_units = []
        self.all_word_units = []
        self.all_tags_in_sequence = []
        self.all_words_in_sequence = []
        self.start_state = "Q0"
        
        
    def get_data(self, input_path):
        
        self.all_tag_units.append(self.start_state)
        with open(input_path, 'r', encoding = 'utf-8') as training_file:
            for sentence in training_file:
                self.all_state_transitions[self.start_state] += 1
                tags_in_current_sequence = []
                words_in_current_sequence = []
                tags_in_current_sequence.append(self.start_state)
                words_in_current_sequence.append("NA")
                tokens_in_current_sequence = sentence.split()
                for i in range(0, len(tokens_in_current_sequence)):
                    current_word, current_tag = tokens_in_current_sequence[i].rsplit('/', 1)
                    if i != len(tokens_in_current_sequence) - 1:
                        self.all_state_transitions[current_tag] += 1
                    tags_in_current_sequence.append(current_tag)
                    words_in_current_sequence.append(current_word)
                    self.all_tag_units.append(current_tag)
                    self.all_word_units.append(current_word)
                    
                self.all_tags_in_sequence.append(tags_in_current_sequence)
                self.all_words_in_sequence.append(words_in_current_sequence)
        
        self.tag_counter = Counter(self.all_tag_units)
        self.word_counter = Counter(self.all_word_units)
        self.unique_tags = list(self.tag_counter.keys())
        self.unique_words = list(self.word_counter.keys())
        
        return self.all_tag_units, self.all_word_units
        
        
    def get_tranistion_probability(self):
        
        self.transition_matrix = defaultdict(dict)
        internal_dict = {}
        
        for tag in self.unique_tags:
        	internal_dict[tag] = 0
        
        for tag in self.unique_tags:
        	self.transition_matrix[tag].update(internal_dict)
        # print(self.transition_matrix)
        
        for i in range(len(self.all_tags_in_sequence)):
        	for j in range(1,len(self.all_tags_in_sequence[i])):
        		self.transition_matrix[self.all_tags_in_sequence[i][j-1]][self.all_tags_in_sequence[i][j]] += 1
    
    
    def add_one_smoothening(self):
        
        for i in self.unique_tags:
            for j in self.unique_tags:
                    self.transition_matrix[i][j] += 1
    
        for i in self.unique_tags:
            for j in self.unique_tags:
                if self.all_state_transitions[i] != 0:
                    self.transition_matrix[i][j] = self.transition_matrix[i][j] / (len(self.unique_tags) + self.all_state_transitions[i])
                else:
                    self.transition_matrix[i][j] = self.transition_matrix[i][j] / len(self.unique_tags)
    
    
    def get_emission_probability(self):
        
        self.emission_matrix = {}
        
        for tag in self.unique_tags:
        	self.emission_matrix[tag] = {}
        
        for i in range(len(self.all_words_in_sequence)):
            for j in range(1,len(self.all_words_in_sequence[i])):
                if self.all_words_in_sequence[i][j] in self.emission_matrix[self.all_tags_in_sequence[i][j]]:
                    self.emission_matrix[self.all_tags_in_sequence[i][j]][self.all_words_in_sequence[i][j]] += 1
                else:
                    self.emission_matrix[self.all_tags_in_sequence[i][j]][self.all_words_in_sequence[i][j]] = 1
    
        for tag in self.unique_tags:
            for word in self.unique_words:
                if tag != self.start_state:
                    if word in self.emission_matrix[tag]:
                        self.emission_matrix[tag][word] = self.emission_matrix[tag][word] / self.tag_counter[tag]
                        
                        
    def generate_output_file(self):
        
        output_file = open("hmmmodel.txt", 'w', encoding='utf-8')
        
        for tag in self.unique_tags:
            output_file.write(str(tag) + "-->")
            for new_tag in self.unique_tags:
                output_file.write(str(new_tag) + "/\::" + str(self.transition_matrix[tag][new_tag]) + " ")
            output_file.write("\n")
        
        output_file.write("***** <<<<< >>>>> *****")
        output_file.write("\n")
        
        for tag in self.unique_tags:
            if tag != self.start_state:
                output_file.write(str(tag) + "-->")
                for new_tag in self.unique_words:
                    if new_tag in self.emission_matrix[tag]:
                        output_file.write(str(new_tag) + "/\::" + str(self.emission_matrix[tag][new_tag]) + " ")
                output_file.write("\n")
        
        output_file.close()
        		
    
if __name__== "__main__":
    
    input_path = 'hmm-training-data/it_isdt_train_tagged.txt'
    dev_path = 'hmm-training-data/it_isdt_dev_tagged.txt'
    hmm_model = HMM_learn()
    hmm_model_2 = HMM_learn()
    train_tags, train_data = hmm_model.get_data(input_path)
    dev_tags, dev_data = hmm_model_2.get_data(dev_path)
    set_td = set(train_data)
    set_dd = set(dev_data)
    unseen_words = []
    counter = 0
    for i in set_dd:
        if i not in set_td:
            unseen_words.append(i)
            counter += 1
    print(counter)
    unseen_word_tags = []
    for i in unseen_words:
        index_dev_data = dev_data.index(i)
        unseen_word_tag = dev_tags[index_dev_data+1]
        unseen_word_tags.append(unseen_word_tag)
    hmm_model.get_tranistion_probability()
    hmm_model.add_one_smoothening()
    hmm_model.get_emission_probability()
    hmm_model.generate_output_file()
    
    