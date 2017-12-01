#####################################
# COMP4980-04 "Introduction to NLP" #
# Dr. Stan Szpakowicz               #
# Thompson Rivers University        #
# Fall 2017                         #
#                                   #
# Project 2                         #
# Chris Kwiatkowski (T00050075)     #
# Iurii Shamkin     (T00036016)     #
#####################################

import os
import re
import nltk
from nltk import bigrams
from nltk.collocations import ngrams

# getting CWD for relative filenames
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def file_input(message):
   path = input(message)
   if os.path.isfile(path):
      return path
   else:
        path = os.path.join(BASE_DIR, path)
        if os.path.isfile(path):
            return path
        else:
            return file_input('Enter an existing file path: ')


# edited trivialTokenizer to NOT divide words like 'don't' into two separate tokens
def trivialTokenizer(text):
    pattern = re.compile(r"\d+|Mr\.|Mrs\.|Dr\.|\b[A-Z]\.|[a-zA-Z_]+-[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+|--|'s|'d|'ll|'m|'re|'ve|[.,:!?;\"'()\[\]&@#-]")
    return(re.findall(pattern, text))


def main():
    # clear the screen
    os.system('clear')
        
    print ("Welcome to our second project created by Iurii Shamkin & Chris Kwiatkowski\nThe theme of the project is emotion analysis")

    # reading text filename/path from user
    user_path_1 = file_input('\nEnter a filename or folder path to a first text: ')
    
    # omitting reading second text and comparing for now
    # user_path_2 = file_input('\nEnter a filename or folder path to a second text: ')

    print ("\nProcessing files...")

    # reading file while replacing new lines and tokenizing into separate sentences
    file = open(user_path_1).read().replace('\n', ' ')
    sentences = nltk.sent_tokenize(file)
    
    # main loop for extracting unigrams, bigrams and trigrams from each sentence
    for sent in sentences:
        text_unigrams = sent_list = trivialTokenizer(sent)
        #text_unigrams = sent_list = nltk.tokenize.word_tokenize(sent)
        text_bigrams = list(bigrams(sent_list))
        text_trigrams = list(ngrams(sent_list,3))

        print(text_unigrams)
        print(text_bigrams)
        print(text_trigrams)
        input()


main()

