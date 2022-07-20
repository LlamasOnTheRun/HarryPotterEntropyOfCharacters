import pytest
import main


def test_entropy_with_word_set_of_three_words():
    wordSet = []
    for i in range(0, 10): wordSet.append("Cats")
    for i in range(0, 14): wordSet.append("Dogs")
    for i in range(0, 7): wordSet.append("Lions")

    entropyValue = main.entropy(main.get_frequency_dist_tracker(wordSet))

    assert entropyValue == 1.529237139029349


def test_list_of_harry_potter_characters():
    harryPotterCharacters = main.get_harry_potter_characters()

    assert harryPotterCharacters.__len__() == 38
    assert harryPotterCharacters.count("Harry") == 1


def test_text_of_harry_potter_philosopher_stone():
    tokenizedCharacters = main.tokenize_harry_potter_book_philosopher_stone()

    assert tokenizedCharacters.__len__() > 0
    assert tokenizedCharacters.count("Harry") > 10
    assert list(set(tokenizedCharacters)).__len__() == 24


def test_text_of_harry_potter_chamber_of_secret():
    tokenizedCharacters = main.tokenize_harry_potter_book_chamber_of_secrets()

    assert tokenizedCharacters.__len__() > 0
    assert tokenizedCharacters.count("Harry") > 10
    assert list(set(tokenizedCharacters)).__len__() == 28


def test_class_freqdisttracker_reverse():
    names = ["Ryan", "Saul", "Jacob", "Jean"]
    probs = [.35, .25, .20, .10]

    freqDistTracker = main.FreqDistTracker(names, probs)
    reversedFreqDistTracker = freqDistTracker.reverse()

    assert freqDistTracker.names.__eq__(names)
    assert freqDistTracker.probs.__eq__(probs)

    names.reverse()
    probs.reverse()
    assert reversedFreqDistTracker.names.__eq__(names) is True
    assert reversedFreqDistTracker.probs.__eq__(probs) is True


def test_class_freqdisttracker_huffman_node_conversion():
    names = ["Ryan", "Saul", "Jacob", "Jean"]
    probs = [.35, .25, .20, .20]

    freqDistTracker = main.FreqDistTracker(names, probs)

    huffmanLeafNodes = freqDistTracker.convert_to_huffman_leaf_nodes()
    huffmanNames = [node.symbol for node in huffmanLeafNodes]
    huffmanProbs = [node.probabilitySum for node in huffmanLeafNodes]

    assert names.__eq__(huffmanNames)
    assert probs.__eq__(huffmanProbs)


def test_huffman_encoding_tree_is_valid_with_descending_frequency():
    names = ["Ryan", "Saul", "Jacob", "Jean", "Katie"]
    probs = [.35, .25, .20, .10, .10]

    freqDistTracker = main.FreqDistTracker(names, probs)
    huffmanTree = main.get_huffman_encoding_tree(freqDistTracker.convert_to_huffman_leaf_nodes())

    assert huffmanTree[0].probabilitySum == 1.0
    assert huffmanTree[0].left.left.probabilitySum == .35
    assert huffmanTree[0].right.right.right.probabilitySum == .10
