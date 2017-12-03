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

# List of negations and intensifiers are borrowed from https://github.com/cjhutto/vaderSentiment

import os
import re
import nltk
from nltk import bigrams
from nltk import FreqDist
from nltk.collocations import ngrams

# This is a subset of stop words from ntlk which an intensifier or negation can not be applied to.
# For example, 'without a word'. When the 2nd token in such a trigram is not one of the words
# in this list, we do not guess that the intensifier applies to the third word but to the second word.
# For example, in 'not because good' the negation applies to because instead of good. However, in a
# trigram like 'not a good', the negation does apply to 'good'.
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
    pattern = re.compile(r"\d+|Mr\.|Mrs\.|Dr\.|\b[A-Z]\.|[a-zA-Z_]+-[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+'t|[a-zA-Z_]+|--|'s|'d|'ll|'m|'re|'ve|[.,:!?;\"()\[\]&@#-]")
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
    user_path_2 = file_input('\nEnter a filename or folder path to a second text: ')

    print ("\nProcessing files...")

    def eval_text(path):
        # reading file while replacing new lines and tokenizing into separate sentences
        file = open(path).read().replace('\n', ' ')
        file = file.lower()
        sentences = nltk.sent_tokenize(file)
        
        # emotion_type => (unnegated_count, negated_count)
        emotion_count = {1: (0,0), 2: (0,0), 3: (0,0), 4: (0,0), 5: (0,0), 6: (0,0), 7: (0,0), 8: (0,0)}
        expression_count = {'total': 0,
                            'unigram': 0, 
                            # Bigrams
                            'n': 0, 'i': 0, 'di': 0, 
                            # Trigrams
                            ('n', 'x'): 0, ('n', 'i'): 0, ('n', 'n'): 0, ('n', 'di'): 0,
                            ('i', 'x'): 0, ('i', 'i'): 0, ('i', 'n'): 0, ('i', 'di'): 0,
                            ('di', 'x'): 0, ('di', 'i'): 0, ('di', 'n'): 0, ('di', 'di'): 0}

        example_expressions = {'unigram': [], 'n': [], 'i': [], 'di': [], ('n', 'x'): [], ('n', 'i'): [], ('n', 'n'): [], ('n', 'di'): [],
                            ('i', 'x'): [], ('i', 'i'): [], ('i', 'n'): [], ('i', 'di'): [],
                            ('di', 'x'): [], ('di', 'i'): [], ('di', 'n'): [], ('di', 'di'): []}

        # main loop for extracting unigrams, bigrams and trigrams from each sentence
        token_count = 0
        sent_count = 0

        for sent in sentences:
            sent_count += 1
            text_unigrams = sent_list = trivialTokenizer(sent)
            
            token_count += len(text_unigrams)  # Get a total count of number of tokens in the text.
            
            text_bigrams = list(bigrams(sent_list))
            text_trigrams = list(ngrams(sent_list,3))

            for unigram in text_unigrams:
                if unigram in emotion_lexicon:      
                    expression_count['unigram'] += 1
                    example_expressions['unigram'].append(unigram)

                    emotion_types = emotion_lexicon[unigram]
                    for e in emotion_types:
                        (unnegated_count, negated_count) = emotion_count[e]
                        emotion_count[e] = (unnegated_count + 1, negated_count)

            # Save the current count of number of tokens that are in the emotion lexicon.
            expression_count['total'] = expression_count['unigram']

            for bigram in text_bigrams:
                if bigram[1] in emotion_lexicon:
                    emotion_types = emotion_lexicon[bigram[1]]

                    if bigram[0] in NEGATIONS_SET:
                        expression_count['unigram'] -= 1  # Subtract from unigram since it was counted previously as unigram
                        expression_count['n'] += 1
                        example_expressions['n'].append(' '.join(bigram))

                        for e in emotion_types: 
                            (unnegated_count, negated_count) = emotion_count[e]
                            emotion_count[e] = (unnegated_count - 1, negated_count + 1)

                    elif bigram[0] in INTENSIFIER_DICT:
                        expression_count['unigram'] -= 1  # Subtract from unigram since it was counted previously as unigram
                        
                        multiplier = INTENSIFIER_DICT[bigram[0]]

                        if multiplier > 1:
                            expression_count['i'] += 1
                            example_expressions['i'].append(' '.join(bigram))
                        else:
                            # Count deintensifiers separately from intensifiers
                            expression_count['di'] += 1
                            example_expressions['di'].append(' '.join(bigram))

                        for e in emotion_types: 
                            (unnegated_count, negated_count) = emotion_count[e]
                            emotion_count[e] = (unnegated_count - 1 + multiplier, negated_count)

            for trigram in text_trigrams:
                if trigram[2] in emotion_lexicon:
                    emotion_types = emotion_lexicon[trigram[2]]
                    
                    word_1 = ''
                    if trigram[0] in NEGATIONS_SET:
                        word_1 = 'n'
                    elif trigram[0] in INTENSIFIER_DICT:
                        word_1 = 'i'
                    elif trigram[0] in STOP_WORDS:
                        word_1 = 'x'

                    word_2 = ''
                    if trigram[1] in NEGATIONS_SET:
                        word_2 = 'n'
                    elif trigram[1] in INTENSIFIER_DICT:
                        word_2 = 'i'
                    elif trigram[1] in STOP_WORDS:
                        word_2 = 'x'

                    if word_1 == 'n' and word_2 == 'x':
                        expression_count[('n', 'x')] += 1
                        example_expressions[('n', 'x')].append(' '.join(trigram))
                        expression_count['unigram'] -= 1

                        for e in emotion_types:
                            (unnegated_count, negated_count) = emotion_count[e]
                            emotion_count[e] = (unnegated_count - 1, negated_count + 1)

                    if word_1 == 'i' and word_2 == 'x':
                        expression_count['unigram'] -= 1

                        multiplier = INTENSIFIER_DICT[trigram[0]]
                        if multiplier > 1:
                            expression_count[('i', 'x')] += 1
                            example_expressions[('i', 'x')].append(' '.join(trigram))

                        else:
                            expression_count[('di', 'x')] += 1
                            example_expressions[('di', 'x')].append(' '.join(trigram))

                        for e in emotion_types:
                            # Subtract the previously counted unnegated unigram, then add the weighted value.
                            # The bigram was not counted since the second word is not a negation nor intensifier.
                            # so the unigram count was not adjusted in the bigram loop.
                            (unnegated_count, negated_count) = emotion_count[e]
                            emotion_count[e] = (unnegated_count - 1 + multiplier, negated_count)

                    if word_1 == 'n' and word_2 == 'n':
                        expression_count[('n', 'n')] += 1
                        example_expressions[('n', 'n')].append(' '.join(trigram))

                        expression_count['n'] -= 1

                        for e in emotion_types:
                            # Subtract the previous negated count as bigram.
                            (unnegated_count, negated_count) = emotion_count[e]
                            emotion_count[e] = (unnegated_count + 1, negated_count - 1)

                    if word_1 == 'i' and word_2 == 'n':
                        expression_count['n'] -= 1

                        multiplier = INTENSIFIER_DICT[trigram[0]]

                        if multiplier > 1:
                            expression_count[('i', 'n')] += 1
                            example_expressions[('i', 'n')].append(' '.join(trigram))

                        else:
                            expression_count[('di', 'n')] += 1
                            example_expressions[('di', 'n')].append(' '.join(trigram))

                        for e in emotion_types:
                            (unnegated_count, negated_count) = emotion_count[e]
                            emotion_count[e] = (unnegated_count, negated_count - 1 + multiplier)

                    if word_1 == 'n' and word_2 == 'i':
                        expression_count['i'] -= 1

                        multiplier = INTENSIFIER_DICT[trigram[1]]

                        if multiplier > 1:
                            expression_count[('n', 'i')] += 1
                            example_expressions[('n', 'i')].append(' '.join(trigram))
                        else:
                            expression_count[('n', 'di')] += 1
                            example_expressions[('n', 'di')].append(' '.join(trigram))


                        for e in emotion_types:
                            # Undo the previous count as a bigram with an intensifier, so subtract multiplier.
                            (unnegated_count, negated_count) = emotion_count[e]
                            emotion_count[e] = (unnegated_count - multiplier, negated_count + multiplier)

                    if word_1 == 'i' and word_2 == 'i':
                        expression_count['i'] -= 1

                        multiplier_1 = INTENSIFIER_DICT[trigram[0]]
                        multiplier_2 = INTENSIFIER_DICT[trigram[1]]

                        type_1 = multiplier_1 > 1 and 'i' or 'di'
                        type_2 = multiplier_2 > 1 and 'i' or 'di'

                        expression_count[(type_1, type_2)] += 1
                        example_expressions[(type_1, type_2)].append(' '.join(trigram))

                        # Since a multiplier of 1 is considered neutral, we need to determine the 
                        # 'direction' of the intensifier, then multiply by the first intensifier
                        # to get the new 'distance' for the second intensifier (the first intensifier affects
                        # the second intensifier not the emotion word). The new 'vector' is then changed back
                        # to where 1 is neutral.
                        new_multiplier = multiplier_1 * (multiplier_2 - 1) + 1

                        for e in emotion_types:
                            # Undo the previous count by subtracting the original multiplier.
                            (unnegated_count, negated_count) = emotion_count[e]
                            emotion_count[e] = (unnegated_count - multiplier_2 + new_multiplier, negated_count)

        print('-----------------------------------------------------------------------------')
        print('Summary for', path)
        print('')
        print('Sentence count:', sent_count)
        print('Total number of tokens:', token_count)

        print('-----------------------------------------------------------------------------')
        print('')
        print('Expression types:')
        print('')


        def get_examples(type, num=2):
            freq = FreqDist(example_expressions[type])
            most_common = freq.most_common(num)
            strings = []
            for (word, word_c) in most_common:
                strings.append('{} ({})'.format(word, word_c))
            return ', '.join(strings)

        print('Total emotion expressions:', expression_count['total'])
        print('')

        print('Unigram count:', expression_count['unigram'])
        print('Most common: {}'.format(get_examples('unigram', 5)))
        print('')

        print('Bigram total count:', expression_count['n'] + expression_count['i'] + expression_count['di'])
        print('Negation      + emotion: {:5.0f}\t{:}'.format(expression_count['n'], get_examples('n', 3)))
        print('Intensifier   + emotion: {:5.0f}\t{:}'.format(expression_count['i'], get_examples('i', 3)))
        print('Deintensifier + emotion: {:5.0f}\t{:}'.format(expression_count['di'], get_examples('di', 3)))
        print('')

        trigram_total = 0
        for k in [('n', 'x'), ('n', 'i'), ('n', 'n'), ('n', 'di'), ('i', 'x'), ('i', 'i'), ('i', 'n'), ('i', 'di'), ('di', 'x'), ('di', 'i'), ('di', 'n'), ('di', 'di')]:
            trigram_total += expression_count[k]

        print('Trigram total count:', trigram_total)
        
        print('Negation      + stop word     + emotion: {:4.0f}\t {:}'.format(expression_count[('n', 'x')], get_examples(('n', 'x'))))
        print('Negation      + negation      + emotion: {:4.0f}\t {:}'.format(expression_count[('n', 'n')], get_examples(('n', 'n'))))
        print('Negation      + intensifier   + emotion: {:4.0f}\t {:}'.format(expression_count[('n', 'i')], get_examples(('n', 'i'))))
        print('Negation      + deintensifier + emotion: {:4.0f}\t {:}'.format(expression_count[('n', 'di')], get_examples(('n', 'di'))))

        print('Intensifier   + stop word     + emotion: {:4.0f}\t {:}'.format(expression_count[('i', 'x')], get_examples(('i', 'x'))))
        print('Intensifier   + negation      + emotion: {:4.0f}\t {:}'.format(expression_count[('i', 'n')], get_examples(('i', 'n'))))
        print('Intensifier   + intensifier   + emotion: {:4.0f}\t {:}'.format(expression_count[('i', 'i')], get_examples(('i', 'i'))))
        print('Intensifier   + deintensifier + emotion: {:4.0f}\t {:}'.format(expression_count[('i', 'di')], get_examples(('i', 'di'))))

        print('Deintensifier + stop word     + emotion: {:4.0f}\t {:}'.format(expression_count[('di', 'x')], get_examples(('di', 'x'))))
        print('Deintensifier + negation      + emotion: {:4.0f}\t {:}'.format(expression_count[('di', 'n')], get_examples(('di', 'n'))))
        print('Deintensifier + intensifier   + emotion: {:4.0f}\t {:}'.format(expression_count[('di', 'i')], get_examples(('di', 'i'))))
        print('Deintensifier + deintensifier + emotion: {:4.0f}\t {:}'.format(expression_count[('di', 'di')], get_examples(('di', 'di'))))

        print('-----------------------------------------------------------------------------')
        print('')
        print('Emotion percentages:')
        print('')

        total = 0
        for (k,v) in emotion_count.items():
            total += v[0] + v[1]

        print("Total weighted sum:", total)

        print('')
        print('emotion (count): %               not emotion (count): %')
        print('')
        percentage_dict = {}
        for (k,v) in emotion_count.items():
            percentage_dict[k] = (v[0]/total*100, v[1]/total*100)
            print("{:12} ({:.0f}): {:.1f}% \t not {:12} ({:.0f}): {:.1f}%".format(emotion_name[k], v[0], v[0]/total*100, emotion_name[k], v[1], v[1]/total*100))

        print('')

        print('Emotion percentages grouped by positive/neutral/negative')
        print('')

        # positive side
        pos_total = (0,0)
        for k in [5,8]:
            (i, j) = emotion_count[k]
            pos_total = (pos_total[0] + i, pos_total[1] + j)

        # neutral side
        neutral_total = (0,0)
        for k in [2,7]:
            (i, j) = emotion_count[k]
            neutral_total = (neutral_total[0] + i, neutral_total[1] + j)

        # negative side
        neg_total = (0,0)
        for k in [1,3,4,6]:
            (i, j) = emotion_count[k]
            neg_total = (neg_total[0] + i, neg_total[1] + j)

        print("{:12} ({:.0f}): {:.1f}% \t negated {:12} ({:.0f}): {:.1f}%".format("positive", pos_total[0], pos_total[0]/total*100, "positive", pos_total[1], pos_total[1]/total*100))
        print("{:12} ({:.0f}): {:.1f}% \t negated {:12} ({:.0f}): {:.1f}%".format("neutral", neutral_total[0], neutral_total[0]/total*100, "neutral", neutral_total[1], neutral_total[1]/total*100))
        print("{:12} ({:.0f}): {:.1f}% \t negated {:12} ({:.0f}): {:.1f}%".format("negative", neg_total[0], neg_total[0]/total*100, "negative", neg_total[1], neg_total[1]/total*100))

        return percentage_dict

    percentage_dict_1 = eval_text(user_path_1)
    print('')
    input('Press enter to evaluate second text...')
    percentage_dict_2 = eval_text(user_path_2)
    print('')
    input('Press enter to see comparison...')

    str_1 = 'The text ' + user_path_1 + ' is more: '
    str_2 = 'The text ' + user_path_2 + ' is more: '

    emotions_1 = []
    emotions_2 = []

    for e in emotion_name:
        (unnegated_1, negated_1) = percentage_dict_1[e]
        (unnegated_2, negated_2) = percentage_dict_2[e]

        if unnegated_1 > unnegated_2:
            emotions_1.append(((unnegated_1 - unnegated_2) / unnegated_2 * 100, '\t{:} (+{:.2f}%)'.format(emotion_name[e], (unnegated_1 - unnegated_2) / unnegated_2 * 100)))
        elif unnegated_1 < unnegated_2:
            emotions_2.append(((unnegated_2 - unnegated_1) / unnegated_1 * 100, '\t{:} (+{:.2f}%)'.format(emotion_name[e], (unnegated_2 - unnegated_1) / unnegated_1 * 100)))

        if negated_1 > negated_2:
            emotions_1.append(((negated_1 - negated_2) / negated_2 * 100, '\t"not {:}" (+{:.2f}%)'.format(emotion_name[e], (negated_1 - negated_2) / negated_2 * 100)))
        elif negated_1 < negated_2:
            emotions_1.append(((negated_2 - negated_1) / negated_1 * 100, '\t"not {:}" (+{:.2f}%)'.format(emotion_name[e],  (negated_2 - negated_1) / negated_1 * 100)))

    print('-----------------------------------------------------------------------------')
    print('Comparison of {} and {}'.format(user_path_1, user_path_2))
    print('')

    if (len(emotions_1) > 0):
        print(str_1)
        print('\n'.join(map(lambda x: x[1], sorted(emotions_1, reverse=True))))
        print('')
    if (len(emotions_2) > 0):
        print(str_2)
        print('\n'.join(map(lambda x: x[1], sorted(emotions_2, reverse=True))))

    print('')

main()
