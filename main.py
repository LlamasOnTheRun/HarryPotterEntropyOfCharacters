import math
import nltk
import matplotlib.pyplot as plt
import requests
import copy


class FreqDistTracker:
    def __init__(self, names, probs):
        self.names = names
        self.probs = probs

    def reverse(self):
        newFreqDistTracker = copy.deepcopy(self)
        newFreqDistTracker.names.reverse()
        newFreqDistTracker.probs.reverse()
        return newFreqDistTracker

    def convert_to_huffman_leaf_nodes(self):
        huffmanNodeList = []
        for i in range(len(self.probs)):
            huffmanNodeList.append(HuffmanNode(self.names[i], self.probs[i]))
        return huffmanNodeList


class HuffmanNode:
    def __init__(self, symbol, probabilitySum):
        self.symbol = symbol
        self.probabilitySum = probabilitySum
        self.left = None
        self.right = None

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right


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


def print_huffman_encodings_for_symbols(huffmanTreeNode, binaryCode):
    if huffmanTreeNode.left is not None:
        print_huffman_encodings_for_symbols(huffmanTreeNode.left, binaryCode + "0")
    if huffmanTreeNode.right is not None:
        print_huffman_encodings_for_symbols(huffmanTreeNode.right, binaryCode + "1")
    if huffmanTreeNode.left is None and huffmanTreeNode.right is None:
        print("Binary for Symbol " + huffmanTreeNode.symbol +
              " with probability " + str(round(huffmanTreeNode.probabilitySum * 100.0, 2)) + "%" +
              " is: " + binaryCode)
    return


def perform_huffman_coding(descendingFreqDistributionTracker):
    huffmanLeafNodes = descendingFreqDistributionTracker.convert_to_huffman_leaf_nodes()
    huffmanTree = build_huffman_encoding_tree(huffmanLeafNodes)
    print_huffman_encodings_for_symbols(huffmanTree[0], "")

    return "listOfEncodingsForEachSymbol"


def build_huffman_encoding_tree(huffmanLeafNodes):
    huffmanTree = copy.copy(huffmanLeafNodes)
    while len(huffmanTree) != 1:
        rightNode = huffmanTree.pop()
        leftNode = huffmanTree.pop()

        newProbabilitySum = leftNode.probabilitySum + rightNode.probabilitySum
        newParentNode = HuffmanNode(str(newProbabilitySum), newProbabilitySum)
        newParentNode.left = leftNode
        newParentNode.right = rightNode

        index = huffmanTree.__len__() - 1
        while index >= 0 and huffmanTree[index].probabilitySum < newParentNode.probabilitySum:
            index = index - 1

        huffmanTree.insert(index + 1, newParentNode)

    return huffmanTree


def main():
    print("******************************* Philosophers Stone Data *******************************")
    frequencyDistTrackerForPhilosopherStone = get_frequency_dist_tracker(tokenize_harry_potter_book_philosopher_stone())
    graph_frequency_dist(frequencyDistTrackerForPhilosopherStone)
    perform_huffman_coding(frequencyDistTrackerForPhilosopherStone)
    print("Entropy of characters from \"Philosophers Stone\": " + str(entropy(frequencyDistTrackerForPhilosopherStone)))

    print("\n******************************* Chamber of Secrets Data *******************************")
    frequencyDistTrackerForChamberOfSecrets = get_frequency_dist_tracker(
        tokenize_harry_potter_book_chamber_of_secrets())
    graph_frequency_dist(frequencyDistTrackerForChamberOfSecrets)
    perform_huffman_coding(frequencyDistTrackerForChamberOfSecrets)
    print("Entropy of characters from \"Chamber of Secrets\": " + str(entropy(frequencyDistTrackerForChamberOfSecrets)))


main()
