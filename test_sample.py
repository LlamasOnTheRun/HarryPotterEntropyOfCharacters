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

    assert harryPotterCharacters.__len__() == 37
    assert harryPotterCharacters.count("Ron") == 1


def test_text_of_ron_philosopher_stone():
    tokenizedCharacters = main.tokenize_harry_potter_book_philosopher_stone()

    assert tokenizedCharacters.__len__() > 0
    assert tokenizedCharacters.count("Ron") > 10
    assert list(set(tokenizedCharacters)).__len__() == 23


def test_text_of_ron_chamber_of_secret():
    tokenizedCharacters = main.tokenize_harry_potter_book_chamber_of_secrets()

    assert tokenizedCharacters.__len__() > 0
    assert tokenizedCharacters.count("Ron") > 10
    assert list(set(tokenizedCharacters)).__len__() == 27


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


def test_class_freqdisttracker_get_prob_of_name():
    names = ["Ryan", "Saul", "Jacob", "Jean"]
    probs = [.35, .25, .20, .20]

    freqDistTracker = main.FreqDistTracker(names, probs)

    assert freqDistTracker.get_prob_of_name("Jacob") == .20
    assert freqDistTracker.get_prob_of_name("Saul") == .25
    assert freqDistTracker.get_prob_of_name("Billy Bob Joe") == 0


def test_huffman_encoding_tree_is_valid_with_descending_frequency():
    names = ["Ryan", "Saul", "Jacob", "Jean", "Katie"]
    probs = [.10, .10, .20, .25, .35]

    freqDistTracker = main.FreqDistTracker(names, probs)
    huffmanTree = main.build_huffman_encoding_tree(freqDistTracker.convert_to_huffman_leaf_nodes())

    assert huffmanTree[0].probabilitySum == 1.0
    assert huffmanTree[0].left.left.probabilitySum == .10
    assert huffmanTree[0].right.left.probabilitySum == .35


def test_craft_probability_matrix():
    print()
    names = ["Ryan", "Saul", "Jacob", "Jean", "Katie"]
    probs = [.35, .25, .20, .10, .10]
    freqDistTracker1 = main.FreqDistTracker(names, probs)

    names = ["Kyle", "Saul", "Bob", "Jean", "Katie", "Kevin", "Mike"]
    probs = [.10, .10, .15, .10, .10, .15, .30]
    freqDistTracker2 = main.FreqDistTracker(names, probs)

    distinctNames = list(set(freqDistTracker1.names + freqDistTracker2.names))
    print(distinctNames)

    probabilityMatrix = main.craft_joint_probability_matrix(distinctNames, freqDistTracker1, freqDistTracker2)

    assert distinctNames.__len__() == 9
    assert probabilityMatrix.matrix.__len__() == distinctNames.__len__()
    for row in probabilityMatrix.matrix: assert row.__len__() == distinctNames.__len__()

    marginalSum = 0
    for value in probabilityMatrix.marginalProbForXAxis: marginalSum = value + marginalSum
    assert pytest.approx(marginalSum) == 1
    print(probabilityMatrix.marginalProbForXAxis)

    marginalSum = 0
    for value in probabilityMatrix.marginalProbForYAxis: marginalSum = value + marginalSum
    assert pytest.approx(marginalSum) == 1
    print(probabilityMatrix.marginalProbForYAxis)
