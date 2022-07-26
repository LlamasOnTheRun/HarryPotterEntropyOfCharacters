import math
import nltk
import matplotlib.pyplot as plt
import requests
import copy
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout, write_dot


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

    def get_prob_of_name(self, name):
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.probs[i]
        return 0


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


def print_graph_for_huffman_tree(huffmanTreeNode, graph):
    graph.add_node(huffmanTreeNode.symbol)
    if huffmanTreeNode.left is not None:
        graph.add_edge(huffmanTreeNode.symbol, huffmanTreeNode.left.symbol)
        print_graph_for_huffman_tree(huffmanTreeNode.left, graph)
    if huffmanTreeNode.right is not None:
        graph.add_edge(huffmanTreeNode.symbol, huffmanTreeNode.right.symbol)
        print_graph_for_huffman_tree(huffmanTreeNode.right, graph)
    if huffmanTreeNode.left is None and huffmanTreeNode.right is None:
        pass
    return


def perform_huffman_coding(descendingFreqDistributionTracker):
    huffmanLeafNodes = descendingFreqDistributionTracker.convert_to_huffman_leaf_nodes()
    huffmanTree = build_huffman_encoding_tree(huffmanLeafNodes)
    print_huffman_encodings_for_symbols(huffmanTree[0], "")
    G = nx.DiGraph()
    print_graph_for_huffman_tree(huffmanTree[0], G)
    write_dot(G, 'test.dot')
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.show()

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


def find_relationship_between_names(fDTForPhilosopherStone, fDTForChamberOfSecrets):
    distinctNames = list(set(fDTForPhilosopherStone.names + fDTForChamberOfSecrets.names))
    numOfDistinctNames = distinctNames.__len__()
    jointProbabilityMatrix = [[0 for x in range(numOfDistinctNames)] for y in range(numOfDistinctNames)]

    marginalProbabilitiesOfXAxisPS = []
    marginalProbabilitiesOfYAxisCOS = []
    xIndex = 0
    yIndex = 0
    for name1 in distinctNames:
        marginalProbabilitiesOfXAxisPS.append(fDTForPhilosopherStone.get_prob_of_name(name1))
        marginalProbabilitiesOfYAxisCOS.append(fDTForChamberOfSecrets.get_prob_of_name(name1))
        yIndex = 0
        for name2 in distinctNames:
            probabilityOfName1InPS = fDTForPhilosopherStone.get_prob_of_name(name1)
            probabilityOfName2InCOS = fDTForChamberOfSecrets.get_prob_of_name(name2)
            jointProbabilityMatrix[xIndex][yIndex] = probabilityOfName1InPS * probabilityOfName2InCOS
            yIndex = yIndex + 1
        xIndex = xIndex + 1

    print(jointProbabilityMatrix)
    print("marginalProbabilitiesOfXAxisPS")
    print(marginalProbabilitiesOfXAxisPS)
    print("marginalProbabilitiesOfYAxisCOS")
    print(marginalProbabilitiesOfYAxisCOS)

    return


def main():

    print("******************************* Philosophers Stone Data *******************************")
    frequencyDistTrackerForPhilosopherStone = get_frequency_dist_tracker(tokenize_harry_potter_book_philosopher_stone())
    print("Entropy of characters from \"Philosophers Stone\": " + str(entropy(frequencyDistTrackerForPhilosopherStone)))
    graph_frequency_dist(frequencyDistTrackerForPhilosopherStone)
    perform_huffman_coding(frequencyDistTrackerForPhilosopherStone)

    print("\n******************************* Chamber of Secrets Data *******************************")
    frequencyDistTrackerForChamberOfSecrets = get_frequency_dist_tracker(tokenize_harry_potter_book_chamber_of_secrets())
    print("Entropy of characters from \"Chamber of Secrets\": " + str(entropy(frequencyDistTrackerForChamberOfSecrets)))
    graph_frequency_dist(frequencyDistTrackerForChamberOfSecrets)
    perform_huffman_coding(frequencyDistTrackerForChamberOfSecrets)

    find_relationship_between_names(frequencyDistTrackerForPhilosopherStone, frequencyDistTrackerForChamberOfSecrets)


main()
