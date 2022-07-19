import math
import nltk
import matplotlib.pyplot as plt
import requests


class FreqDistTracker:
    def __init__(self, names, probs):
        self.names = names
        self.probs = probs


def entropy(freqDistributionTracker):
    return -sum(p * math.log(p, 2) for p in freqDistributionTracker.probs)  # Calculates entropy for set


def get_frequency_dist_tracker(names):
    freqDistOfNames = nltk.FreqDist(names)  # Will count symbols in a set
    names = [name for name in freqDistOfNames]
    probs = [freqDistOfNames.freq(name) for name in freqDistOfNames]  # Will calculate the probability of said count
    return FreqDistTracker(names, probs)


def graph_frequency_dist(freqDistributionTracker):
    plt.bar(freqDistributionTracker.names, freqDistributionTracker.probs)
    plt.xticks(rotation=60)
    plt.show()


def tokenize_harry_potter_book_philosopher_stone():
    url = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%201" \
          "%20-%20The%20Philosopher's%20Stone.txt"
    bookText = requests.get(url).text

    removePunctuationTokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    bookTextPunctuationRemoved = removePunctuationTokenizer.tokenize(bookText)

    getCharacterNamesTokenizer = nltk.tokenize.RegexpTokenizer(get_regex_for_all_characters())
    return getCharacterNamesTokenizer.tokenize(' '.join(bookTextPunctuationRemoved))


def tokenize_harry_potter_book_chamber_of_secrets():
    url = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%202" \
          "%20-%20The%20Chamber%20of%20Secrets.txt"
    bookText = requests.get(url).text

    removePunctuationTokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    bookTextPunctuationRemoved = removePunctuationTokenizer.tokenize(bookText)

    getCharacterNamesTokenizer = nltk.tokenize.RegexpTokenizer(get_regex_for_all_characters())
    return getCharacterNamesTokenizer.tokenize(' '.join(bookTextPunctuationRemoved))


def get_harry_potter_characters():
    harryPotterCharacterNamesFile = open("harry_potter_characters.txt", "r")
    return harryPotterCharacterNamesFile.read().splitlines()


def get_regex_for_all_characters():
    regexForAllCharacterNames = ""
    for character in get_harry_potter_characters():
        regexForAllCharacterNames += character + "|"
    return regexForAllCharacterNames[:-1]


#def huffman_encoding(names, probabilities):
#    return listOfEncodingsForEachSymbol


def main():
    frequencyDistTrackerForPhilosopherStone = get_frequency_dist_tracker(tokenize_harry_potter_book_philosopher_stone())
    graph_frequency_dist(frequencyDistTrackerForPhilosopherStone)
    print("Entropy of characters from \"Philosophers Stone\": " + str(entropy(frequencyDistTrackerForPhilosopherStone)))

    frequencyDistTrackerForChamberOfSecrets = get_frequency_dist_tracker(tokenize_harry_potter_book_chamber_of_secrets())
    graph_frequency_dist(frequencyDistTrackerForChamberOfSecrets)
    print("Entropy of characters from \"Chamber of Secrets\": " + str(entropy(frequencyDistTrackerForChamberOfSecrets)))


main()
