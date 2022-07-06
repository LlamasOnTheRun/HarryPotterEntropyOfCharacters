import math
import nltk

def entropy(labels):
    freqdist = nltk.FreqDist(labels) # Will count symbols in a set
    probs = [freqdist.freq(l) for l in freqdist] # Will calculate the probability of said count
    return -sum(p * math.log(p,2) for p in probs) # Calculates entropy for set

wordSet = []
for i in range(0,10): wordSet.append("Cats")
for i in range(0,14): wordSet.append("Dogs")
for i in range(0,7): wordSet.append("Lions")

print(entropy(wordSet))