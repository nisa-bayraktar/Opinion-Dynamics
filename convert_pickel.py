import pickle
import pprint
import json

my_dict_final = {}  # Create an empty dictionary
with open('new_choice_network_0.03/new_1000_network_choice_1', 'rb') as f:
    my_dict_final.update(pickle.load(f))   # Update contents of file1 to the dictionary
with open('new_choice_network_0.03/new_1000_network_choice_2', 'rb') as f:
    my_dict_final.update(pickle.load(f))   # Update contents of file2 to the dictionary
print (my_dict_final)