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
