import math
import nltk
import matplotlib.pyplot as plt
import requests


def entropy(labels):
    freqdist = nltk.FreqDist(labels)  # Will count symbols in a set
    probs = [freqdist.freq(l) for l in freqdist]  # Will calculate the probability of said count
    plt.bar(freqdist.keys(), probs)
    plt.show()
    return -sum(p * math.log(p, 2) for p in probs)  # Calculates entropy for set


def tokenize_harry_potter_book_philosopher_stone():
    url = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%201" \
          "%20-%20The%20Philosopher's%20Stone.txt"
    response = requests.get(url)
    tokenizer = nltk.tokenize.RegexpTokenizer(get_regex_for_all_characters())
    return tokenizer.tokenize(response.text)


def tokenize_harry_potter_book_chamber_of_secrets():
    url = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%202" \
          "%20-%20The%20Chamber%20of%20Secrets.txt"
    response = requests.get(url)
    tokenizer = nltk.tokenize.RegexpTokenizer(get_regex_for_all_characters())
    return tokenizer.tokenize(response.text)


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
