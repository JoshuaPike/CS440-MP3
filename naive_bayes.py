# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import math
"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=1.0, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set

    # Need P(type = positive), P(type = negative), p(word | type = positive), and P(word | type = negative)
    # alpha b/w 0 and 1 (typically small), all unknown words represented by UNK
    # V is number of word types in training data, num of unique words (I think in all data not just for C)
    # n = num words in type C training data (eg positive), count(W) = num of times W appeared in type C
    # P(UNK | C) = alpha / (n + alpha*(V + 1))
    # P(W | C) = (count(W) + alpha) / (n + alpha*(V + 1))

    # P(type = positive), P(type = negative) easy -----> count(pos)/total, count(neg)/total... As we iterate thru training set, keep count of number of pos/neg 
    # ABOVE IS GIVEN P(type = negative) = 1 - pos_prior
    # Pos/neg dict. Word as key, count as value. iterate thru training data adding to each
    
    posDict = {}
    negDict = {}
    nPos = 0
    nNeg = 0
    ctr = 0
    for email in train_set:
        if train_labels[ctr] == 1:
            for word in email:
                nPos+=1
                if word not in posDict:
                    posDict[word] = 1
                else:
                    posDict[word] = posDict[word] + 1
        else:
            for word in email:
                nNeg+=1
                if word not in negDict:
                    negDict[word] = 1
                else:
                    negDict[word] = negDict[word] + 1
        ctr+=1
    
    Vpos = len(posDict)
    Vneg = len(negDict)

    posProb = {}
    negProb = {}

    for word in posDict.keys():
        posProb[word] = (posDict[word] + smoothing_parameter)/(nPos + smoothing_parameter*(Vpos+1))
        print(posProb[word])
    posProb['UNK'] = smoothing_parameter/(nPos + smoothing_parameter*(Vpos+1))

    for word in negDict.keys():
        negProb[word] = (negDict[word] + smoothing_parameter)/(nNeg + smoothing_parameter*(Vneg+1))
    negProb['UNK'] = smoothing_parameter/(nNeg + smoothing_parameter*(Vneg+1))

    # Now apply to dev set
    dev_labels = []
    for email in dev_set:
        evPos = 0
        evNeg = 0
        for word in email:
            if word in posProb:
                evPos += math.log(posProb[word], 10)
            else:
                evPos += math.log(posProb['UNK'], 10)
            if word in negProb:
                evNeg += math.log(negProb[word], 10)
            else:
                evNeg += math.log(negProb['UNK'], 10)
        totalPos = math.log(pos_prior, 10) + evPos
        totalNeg = math.log(1 - pos_prior, 10) + evNeg
        if (totalPos > totalNeg):
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    return dev_labels

def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=1.0, bigram_smoothing_parameter=1.0, bigram_lambda=0.5,pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smoothing_parameter - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set using a bigram model
    bigram_smoothing_parameter = 0.8
    unigram_smoothing_parameter = 0.7
    posPairDict = {}
    negPairDict = {}
    nPairPos = 0
    nPairNeg = 0
    ctr = 0
    for email in train_set:
        if train_labels[ctr] == 1:
            for i in range(len(email)-1):
                nPairPos+=1
                pair = (email[i], email[i+1])
                if pair not in posPairDict:
                    posPairDict[pair] = 1
                else:
                    posPairDict[pair] = posPairDict[pair]+1
        else:
            for i in range(len(email)-1):
                nPairNeg+=1
                pair = (email[i], email[i+1])
                if pair not in negPairDict:
                    negPairDict[pair] = 1
                else:
                    negPairDict[pair] = negPairDict[pair]+1
        ctr+=1

    VPairpos = len(posPairDict)
    VPairneg = len(negPairDict)
    V = len(set(posPairDict) - set(negPairDict)) + len(set(negPairDict) - set(posPairDict))

    posPairProb = {}
    negPairProb = {}

    for pair in posPairDict.keys():
        posPairProb[pair] = (posPairDict[pair] + bigram_smoothing_parameter)/(nPairPos + bigram_smoothing_parameter*(V+1))
    posPairProb['UNK'] = bigram_smoothing_parameter/(nPairPos + bigram_smoothing_parameter*(V+1))

    for pair in negPairDict.keys():
        negPairProb[pair] = (negPairDict[pair] + bigram_smoothing_parameter)/(nPairNeg + bigram_smoothing_parameter*(V+1))
    negPairProb['UNK'] = bigram_smoothing_parameter/(nPairNeg + bigram_smoothing_parameter*(V+1))

    # Bigram portion is handled, now put in unigram
    posDict = {}
    negDict = {}
    nPos = 0
    nNeg = 0
    ctr = 0
    for email in train_set:
        if train_labels[ctr] == 1:
            for word in email:
                nPos+=1
                if word not in posDict:
                    posDict[word] = 1
                else:
                    posDict[word] = posDict[word] + 1
        else:
            for word in email:
                nNeg+=1
                if word not in negDict:
                    negDict[word] = 1
                else:
                    negDict[word] = negDict[word] + 1
        ctr+=1
    
    Vpos = len(posDict)
    Vneg = len(negDict)
    V2 = len(set(posDict) - set(negDict)) + len(set(negDict) - set(posDict))

    posUniProb = {}
    negUniProb = {}

    for word in posDict.keys():
        posUniProb[word] = (posDict[word] + unigram_smoothing_parameter)/(nPos + unigram_smoothing_parameter*(V2+1))
    posUniProb['UNK'] = unigram_smoothing_parameter/(nPos + unigram_smoothing_parameter*(V2+1))

    for word in negDict.keys():
        negUniProb[word] = (negDict[word] + unigram_smoothing_parameter)/(nNeg + unigram_smoothing_parameter*(V2+1))
    negUniProb['UNK'] = unigram_smoothing_parameter/(nNeg + unigram_smoothing_parameter*(V2+1))

    # Unigram portion is handled, now combine to get dev_labels
    dev_labels = []

    for email in dev_set:
        evUniPos = 0
        evUniNeg = 0
        evPairPos = 0
        evPairNeg = 0
        
        for i in range(len(email)-1):
            word = email[i]
            pair = (email[i], email[i+1])
            if word in posUniProb:
                evUniPos += math.log(posUniProb[word], 10)
            else:
                evUniPos += math.log(posUniProb['UNK'], 10)
            if word in negUniProb:
                evUniNeg += math.log(negUniProb[word], 10)
            else:
                evUniNeg += math.log(negUniProb['UNK'], 10)

            if pair in posPairProb:
                evPairPos += math.log(posPairProb[pair], 10)
            else:
                evPairPos += math.log(posPairProb['UNK'], 10)
            if pair in negPairProb:
                evPairNeg += math.log(negPairProb[pair], 10)
            else:
                evPairNeg += math.log(negPairProb['UNK'], 10)
        # Add last word to pos/neg uni
        if email[len(email)-1] in posUniProb:
            evUniPos += math.log(posUniProb[email[len(email)-1]], 10)
        else:
            evUniPos += math.log(posUniProb['UNK'], 10)
        if email[len(email)-1] in negUniProb:
            evUniNeg += math.log(negUniProb[email[len(email)-1]], 10)
        else:
            evUniNeg += math.log(negUniProb['UNK'], 10)

        # Add everything up
        totalUniPos = evUniPos + math.log(pos_prior, 10)
        totalUniNeg = evUniNeg + math.log(1-pos_prior, 10)
        totalBiPos = evPairPos + math.log(pos_prior, 10)
        totalBiNeg = evPairNeg + math.log(1-pos_prior, 10)

        totalPos = (1-bigram_lambda)*totalUniPos + bigram_lambda*totalBiPos
        totalNeg = (1-bigram_lambda)*totalUniNeg + bigram_lambda*totalBiNeg
        if (totalPos > totalNeg):
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    return dev_labels