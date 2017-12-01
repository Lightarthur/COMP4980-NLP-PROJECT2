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

'''
List of negations and intensifiers are borrowed from https://github.com/cjhutto/vaderSentiment

'''

import os
import re
import nltk
from nltk import bigrams
from nltk.collocations import ngrams

NEGATIONS_LIST = \
["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt", "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't", "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere", "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent", "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't", "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

INTERNSIFIERS_DICT = \
{"absolutely": 1.5, "amazingly": 1.5, "awfully": 1.5, "completely": 1.5, "considerably": 1.5,
 "decidedly": 1.5, "deeply": 1.5, "effing": 1.5, "enormously": 1.5,
 "entirely": 1.5, "especially": 1.5, "exceptionally": 1.5, "extremely": 1.5,
 "fabulously": 1.5, "flipping": 1.5, "flippin": 1.5,
 "fricking": 1.5, "frickin": 1.5, "frigging": 1.5, "friggin": 1.5, "fully": 1.5, "fucking": 1.5,
 "greatly": 1.5, "hella": 1.5, "highly": 1.5, "hugely": 1.5, "incredibly": 1.5,
 "intensely": 1.5, "majorly": 1.5, "more": 1.5, "most": 1.5, "particularly": 1.5,
 "purely": 1.5, "quite": 1.5, "really": 1.5, "remarkably": 1.5,
 "so": 1.5, "substantially": 1.5,
 "thoroughly": 1.5, "totally": 1.5, "tremendously": 1.5,
 "uber": 1.5, "unbelievably": 1.5, "unusually": 1.5, "utterly": 1.5,
 "very": 1.5,
 "almost": 0.5, "barely": 0.5, "hardly": 0.5, "just enough": 0.5,
 "kind of": 0.5, "kinda": 0.5, "kindof": 0.5, "kind-of": 0.5,
 "less": 0.5, "little": 0.5, "marginally": 0.5, "occasionally": 0.5, "partly": 0.5,
 "scarcely": 0.5, "slightly": 0.5, "somewhat": 0.5,
 "sort of": 0.5, "sorta": 0.5, "sortof": 0.5, "sort-of": 0.5}

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

