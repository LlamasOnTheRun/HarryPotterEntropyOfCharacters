import math
import nltk
import matplotlib.pyplot as plt
import requests


def entropy(names):
    freqDistOfNames = nltk.FreqDist(names)  # Will count symbols in a set
    names = [name for name in freqDistOfNames]
    probs = [freqDistOfNames.freq(name) for name in freqDistOfNames]  # Will calculate the probability of said count

    plt.bar(names, probs)
    plt.xticks(rotation=60)
    plt.show()

    return -sum(p * math.log(p, 2) for p in probs)  # Calculates entropy for set


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


def main():
    print(entropy(tokenize_harry_potter_book_philosopher_stone()))
    print(entropy(tokenize_harry_potter_book_chamber_of_secrets()))


main()
