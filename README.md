# COMP4980-NLP-PROJECT2

Project 2

posted on November 15, 2017, due on December 1 at 23:59
by email to sszpakowicz@tru.ca

This work is worth up to 22.5% of the total mark available in the course. Bonus is entirely possible.

Late work will be accepted only till Tuesday December 5, 2017 at 23:59. There will be a penalty of 0.75 (of 22.5) a day.

The project is designed for teams of two. The work of people who prefer to go it alone will be marked in the same way as team work.

Feel free to borrow from the Internet -- but not from each other -- and make sure to credit any such borrowings in comments in your Python code. (I am rather convinced that you will not find anything suitable on the Internet, but YMMV.)

I will run all submitted programs. I would also like every team to talk to me in person, and explain their project. If you prefer to submit documentation with the same explanation, be my guest, but I may still want to ask a question or two.


The theme of the project is emotion analysis, very simple but not impossibly so. The project takes up where homework 1 left off.

I will not bother you here with a lecture on this exciting application of NLP. Let us be practical. The archive at <sites.google.com/site/comp498004/homework/L8.zip> contains, in separate files, words associated with several basic emotions. These lists will be your principal resource.

The task in homework 1 was too simplistic to be noteworthy, but the main idea is still defensible. One must consider valency shifters --- those little negation words, such as "not" or "never", which you looked for.

Just as one counts polarity words, positive and negative, one can count emotion words. Such a word occurs:

- alone, not preceded by a valency shifter;

- or in a bigram, preceded by a negation word (or maybe even two!);

- or in a trigram --- like a bigram but with an intensifier (a strengthening or weakening modifier such as "very", "a little", "rather", "much", and so on) in the middle.

Here comes the twist. While polarity values are naturally opposite (not positive/negative means negative/positive), there is no such opposition for the emotions which the words on those lists express. For example, "not sad" does not mean "joyful".

The effect is a little unsettling: you will be noting twice the number of emotions, with and without a negating prefix. So, not merely two classes, but sixteen.

I better tell you right now that the lists have a significant overlap. Suffice it to say that the *average* pairwise overlap is 222.5 words. (There may be ways around this repetition, but that would be a very different project. Just take this dictionary of emotion words at face value.) This means that you should count exhaustively. For example, the word "sonnet" is both joyful and sad, so if you see it, increase two counts.

[Trivia: the words "feeling" and "treat" are the most ambivalent, each with six appearances; "death", "epidemic", obliging", "opera" and "weight" put in five appearances each.]


As your first task in this project, create a reasonable list of valency shifters and intensifiers, and try to figure out which belong with which: maybe not every combination works.

Your program will analyze (just count, really) expressions of emotion in a text, and apply this analysis in a comparison between two novels from our collection at <sites.google.com/site/comp498004/data>. Test your program on any Dickens/Eliot/Gaskell novels you fancy so long as they are novels of 350KB or more, not novelettes. I will test your program on anything that strikes my fancy, so make sure not to hard-wire those file names.

The counting must be done cleverly, that is to say, by sentence. The result for a novel will be a list of sixteen counts.

Now, for the comparison... I would like to leave it to you to decide how to do it. I need the statistics of each novel, including the percentage of emotion expressions in the text, the counts of emotions expression types (unigram/bigram/trigram, maybe more if you consider more), and so on. Then come the sixteen numbers and their interpretation: which novel is (for what it's worth) more trusting, more suprised, angrier and so on. One can also bundle emotions a little: joy and trust are on the positive side, anticipation and surprise are kind of neutral, and the other four are on the negative side. This can also be evaluated and reported.


I have not asked you yet to pay much attention to intensifiers, other than to consider them inside trigrams. They can, however, be useful. An emotion expression has certain strength. Suppose that every lonely emotion word counts as 1. If preceded by "very", "extremely" and so on, it may count for more, maybe 1.5; note that no negation is allowed before that intensifier, because that would be a different emotion. If preceded by "little", "hardly any" and so on, an emotion expression may count as 0.5.

To make a long story short: if you consider intensifiers in counting, you will get a bonus. Just be sure to think it through: which intensifiers belong with which emotions.

You will also get a bonus if you compare the whole novelistic oeuvre of two authors.


You will submit two text files: a program in Python and a brief document with your findings for the novels you worked with. Do not (do not) send me the texts of the novels.
 

Good luck!