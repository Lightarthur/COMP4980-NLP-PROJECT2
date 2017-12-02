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

STOP_WORDS = {'has', 'into', 'theirs', 'its', 
              'being', 'from', 'have', 'were', 'at', 'with', 'my', 'about', 
              'should', 'did', 'for', 'her', 'their', 'does', 'up', 'had',
              'yours', 'themselves', 'of', 'been', 'by', 'do', 'where', 'against', 'there', 
              'which', 'through', 'in', 'other', 'doing',  
              'those', 'your', 'was', 'hers', 
              'yourself', 'is', 'his', 
              'herself', 'an', 'himself', 'you', 
              'this', 'any', 'the', 'myself', 'ourselves', 'itself', 'that', 'on', 'ours', 
              'just', 'having', 'are', 'our', 'as', 'they', 'to', 
              'these', 'both', 'be', 'them', 'yourselves', 'a'}


NEGATIONS_SET = \
{"couldn't", 'nope', "wouldn't", 'hasnt', 'never', "daren't", 'nowhere', 'rarely', "wasn't", 'without', 'neither', 'wouldnt', 'darent', "mustn't", "won't", "haven't", 'arent', 'havent', 'didnt', 'dont', 'wont', 'doesnt', 'couldnt', 'shouldnt', "shouldn't", "shan't", 'wasnt', "aren't", "mightn't", 'cannot', 'werent', 'hadnt', "don't", "weren't", 'not', 'isnt', "hasn't", 'none', "doesn't", "didn't", 'oughtnt', 'shant', 'aint', "needn't", 'uhuh', "oughtn't", "ain't", 'cant', 'despite', 'seldom', 'mightnt', 'neednt', 'uh-uh', 'mustnt', 'nor', "hadn't", "isn't", "can't", 'nothing'}

INTENSIFIER_DICT = \
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
 "almost": 0.5, "barely": 0.5, "hardly": 0.5,
 "kind of": 0.5, "kinda": 0.5, "kindof": 0.5, "kind-of": 0.5,
 "less": 0.5, "little": 0.5, "marginally": 0.5, "occasionally": 0.5, "partly": 0.5, "rather": 1.25,
 "scarcely": 0.5, "slightly": 0.5, "somewhat": 0.5,
 "sort of": 0.5, "sorta": 0.5, "sortof": 0.5, "sort-of": 0.5, "too": 1.5, "such": 1.5}

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


    emotion_name = {1: 'anger', 2: 'anticipation', 3: 'disgust', 4: 'fear', 5: 'joy', 6: 'sadness', 7: 'surprise', 8: 'trust'}
    emotion_lexicon = {};  # 'word' => [emotion_type] where emotion_type is int in [1,8]

    emotion_lexicon_files = [(1, 'anger.txt'), 
                             (2, 'anticipation.txt'), 
                             (3, 'disgust.txt'), 
                             (4, 'fear.txt'),
                             (5, 'joy.txt'), 
                             (6, 'sadness.txt'), 
                             (7, 'surprise.txt'), 
                             (8, 'trust.txt')]
    for (emotion, lex_file) in emotion_lexicon_files:
        for word in open(lex_file).read().split('\n'):
            if (word == ''): continue
            if (emotion_lexicon.get(word) == None):
                emotion_lexicon[word] = []
            emotion_lexicon[word].append(emotion)
            
    # reading text filename/path from user
    user_path_1 = file_input('\nEnter a filename or folder path to a first text: ')

    # omitting reading second text and comparing for now
    # user_path_2 = file_input('\nEnter a filename or folder path to a second text: ')

    print ("\nProcessing files...")

    # reading file while replacing new lines and tokenizing into separate sentences
    file = open(user_path_1).read().replace('\n', ' ')
    file = file.lower()
    sentences = nltk.sent_tokenize(file)
    
    # emotion_type => (unnegated_count, negated_count)
    emotion_count = {1: (0,0), 2: (0,0), 3: (0,0), 4: (0,0), 5: (0,0), 6: (0,0), 7: (0,0), 8: (0,0)}

    # main loop for extracting unigrams, bigrams and trigrams from each sentence
    for sent in sentences:
        text_unigrams = sent_list = trivialTokenizer(sent)
        #text_unigrams = sent_list = nltk.tokenize.word_tokenize(sent)
        text_bigrams = list(bigrams(sent_list))
        text_trigrams = list(ngrams(sent_list,3))

        
        for unigram in text_unigrams:
            if unigram in emotion_lexicon:
                emotion_types = emotion_lexicon[unigram]
                for e in emotion_types:
                    (unnegated_count, negated_count) = emotion_count[e]
                    emotion_count[e] = (unnegated_count + 1, negated_count)


        for bigram in text_bigrams:
            if bigram[1] in emotion_lexicon:
                emotion_types = emotion_lexicon[bigram[1]]
                for e in emotion_types: 
                    if bigram[0] in NEGATIONS_SET:
                        # print(bigram)
                        (unnegated_count, negated_count) = emotion_count[e]
                        emotion_count[e] = (unnegated_count - 1, negated_count + 1)

                    elif bigram[0] in INTENSIFIER_DICT:
                        multiplier = INTENSIFIER_DICT[bigram[0]]
                        (unnegated_count, negated_count) = emotion_count[e]
                        emotion_count[e] = (unnegated_count - 1 + multiplier, negated_count)

        for trigram in text_trigrams:
            if trigram[2] in emotion_lexicon:
                emotion_types = emotion_lexicon[trigram[2]]
                for e in emotion_types:

                    if trigram[0] in NEGATIONS_SET:
                        if trigram[1] in STOP_WORDS:
                            print('neg x e', trigram)

                        # print(trigram)
                        (unnegated_count, negated_count) = emotion_count[e]
                        emotion_count[e] = (unnegated_count - 1, negated_count + 1)

                    elif trigram[0] in INTENSIFIER_DICT:
                        if trigram[1] in STOP_WORDS:
                            print('int x e', trigram)

                        multiplier = INTENSIFIER_DICT[trigram[0]]
                        (unnegated_count, negated_count) = emotion_count[e]
                        emotion_count[e] = (unnegated_count - 1 + multiplier, negated_count)

        # print(text_unigrams)
        # print(text_bigrams)
        # print(text_trigrams)
        # input()

    print(emotion_count)


main()

