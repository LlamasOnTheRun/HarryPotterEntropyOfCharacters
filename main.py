import math
import nltk
import matplotlib.pyplot as plt


def entropy(labels):
    freqdist = nltk.FreqDist(labels)  # Will count symbols in a set
    probs = [freqdist.freq(l) for l in freqdist]  # Will calculate the probability of said count
    plt.bar(freqdist.keys(), probs)
    return -sum(p * math.log(p, 2) for p in probs)  # Calculates entropy for set


def tokenize_harry_potter_book_philosopher_stone_for_character_names():
    return


def tokenize_harry_potter_book_chamber_of_secrets_for_character_names():
    return


def get_harry_potter_characters():
    harryPotterCharacterNamesFile = open("harry_potter_characters.txt", "r")
    return harryPotterCharacterNamesFile.read().splitlines()


def main():
    plt.show()

# main()
