"""
@file NLP_nltk.py
@brief Natural Language Processing (NLP) using nltk (Natural Language Toolkit)

@author Ernest Yeung
@email ernestyalumni dot gmail dot com

Remember to import these dependencies:
import nltk
import networkx as nx
import itertools
"""  
import nltk
import networkx as nx
import itertools

import string

def unique_everseen(iterable,key=None):
	""" List unique elements in order of appearance. 
	
	Examples:
	unique_everseen('AAAABBBCCDAABBB') --> A B C D 
	unique_everseen('ABBCcAD', str.lower) --> A B C D
	"""
	seen = set()
	seen_add = seen.add
	if key is None:
		for element in [x for x in iterable if x not in seen]:
			seen_add(element)
			yield element
	else:
		for element in iterable:
			k = key(element)
			if k not in seen:
				seen_add(k)
				yield element

def levenshtein_distance(first,second):
	""" Return the levenshtein distance between two strings.  
	
	http://rosettacode.org/wiki/Levenshtein_distance#Python
	"""
	if len(first) > len(second):
		first, second = second, first
	distances = range(len(first)+1)
	for index2, char2 in enumerate(second):
		new_distances = [index2 + 1]
		for index1, char1 in enumerate(first):
			if char1 == char2:
				new_distances.append(distances[index1])
			else:
				new_distances.append(1 + min((distances[index1], 
												distances[index1+1],
													new_distances[-1])))
		distances = new_distances
	return distances[-1]
	

	
				
				
def txtfile_preprocess_basic(filename,defaulttags=['NN','JJ','NNP']):
	f=open(filename)
	fread=f.read()
	fread=fread.decode('utf-8')
	word_tokens = nltk.word_tokenize(fread)

	# Assign POS tags to the words in the text
	tagged = nltk.pos_tag(word_tokens)

	textlist = [x[0] for x in tagged]

	# filter for tags
	tagged_filtered = [item for item in tagged if item[1] in defaulttags]

	# Normalize - return a list of tuples with the first item's periods removed
	tagged_filtered = [(item[0].replace('.',''),item[1]) for item in tagged_filtered]

	unique_word_set=unique_everseen([x[0] for x in tagged_filtered])
	word_set_list=list(unique_word_set)

	return word_set_list, textlist
	
def build_graph(word_set_list):
	
	# Initialize an undirected graph.
	gr = nx.Graph()
	gr.add_nodes_from(word_set_list)
	nodePairs = list(itertools.combinations(word_set_list,2))
	
	for pair in nodePairs:
		firstString = pair[0]
		secondString = pair[1]
		levDistance = levenshtein_distance(firstString, secondString)
		gr.add_edge(firstString,secondString,weight=levDistance)
		
	return gr
	
def do_pagerank(gr, word_set_list,textlist,rationumber=3):
	
	# initial value of 1.0, error tolerance of 0.0001
	
	calculated_page_rank = nx.pagerank(gr, weight='weight')
	
	# Most important words in ascending order of importance
	
	keyphrases = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)
	
	# The number of keyphrases returned will be relative to the size of the text 
	# default (a third of the number of vertices)
	ratioval = len(word_set_list) // rationumber
	keyphrases_ratio = keyphrases[0:ratioval+1]
	
	# Take keyphrases with multiple words into consideration; 
	# if 2 words are adjacent in the text and are selected as keywords, join them together
	modified_key_phrases = set([])
	
	# Keeps track of individual keywords that have been joined to form a keyphrase
	dealt_with = set([])
	
	i=0
	j=1
	while j < len(textlist):
		first  = textlist[i]
		second = textlist[j]
		if first in keyphrases_ratio and second in keyphrases_ratio:
			keyphrase = first + ' ' + second 
			modified_key_phrases.add(keyphrase)
			dealt_with.add(first)
			dealt_with.add(second)
		else:
			if first in keyphrases_ratio and first not in dealt_with:
				modified_key_phrases.add(first)
            
			# if this is the last word in the text, and it is a keyword, it
			# definitely has no chance of being a keyphrase at this point
			if j == len(textlist) - 1 and second in keyphrases_ratio and second not in dealt_with:
				modified_key_phrases.add(second)
        
		i = i + 1
		j = j + 1
	return modified_key_phrases, calculated_page_rank
			

def txt_preprocess(txt,defaulttags=['NN','JJ','NNP','CD','NNS','VB','VBN']):
	filtered_tokens=[i for i in nltk.word_tokenize(txt) if i not in string.punctuation]

	# Assign POS tags to the words in the text
	tagged = nltk.pos_tag(filtered_tokens)

	textlist = [x[0] for x in tagged]

	# filter for tags
	tagged_filtered = [item for item in tagged if item[1] in defaulttags]

	# Normalize - return a list of tuples with the first item's periods removed
	tagged_filtered = [(item[0].replace('.',''),item[1]) for item in tagged_filtered]

	unique_word_set=unique_everseen([x[0] for x in tagged_filtered])
	word_set_list=list(unique_word_set)

	return word_set_list, textlist			
	
	
	
	
	 	
	
