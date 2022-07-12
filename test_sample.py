import pytest
import main


def test_entropy_with_word_set_of_three_words():
    wordSet = []
    for i in range(0, 10): wordSet.append("Cats")
    for i in range(0, 14): wordSet.append("Dogs")
    for i in range(0, 7): wordSet.append("Lions")

    entropyValue = main.entropy(wordSet)

    assert entropyValue == 1.529237139029349


def test_list_of_harry_potter_characters():

    harryPotterCharacters = main.get_harry_potter_characters()

    assert harryPotterCharacters.__sizeof__() > 0
    assert harryPotterCharacters.count("Harry") == 1
