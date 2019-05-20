import os

import nltk

from rake import Rake

all_phrases = []

with open("./test_documents/high_frequency.txt", "r") as f:
    lines = f.readlines()
    input_text = "".join(lines)

with open("./test_documents/dementia_plaques.txt", "r") as f:
    lines = f.readlines()
    input_text2 = "".join(lines)


def get_phrases():
    r = Rake()
    phrases = r.get_phrases(input_text)
    all_phrases.append([phrase[0] for phrase in phrases])


def get_phrases2():
    r = Rake()
    phrases = r.get_phrases(input_text2)
    all_phrases.append([phrase[0] for phrase in phrases])


if __name__ == "__main__":
    import time

    start = time.time()
    get_phrases()
    get_phrases2()
    print(f"Execution time: {time.time()-start}")
