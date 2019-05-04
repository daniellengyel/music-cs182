import gensim
from gensim.models.keyedvectors import KeyedVectors
import pickle
import sys


def report_progress(progress, total, lbar_prefix = '', rbar_prefix=''):
    """
    Create progress bar for large task
    """
    percent = round(progress / float(total) * 100, 2)
    buf = "{0}|{1}| {2}{3}/{4} {5}%".format(lbar_prefix, ('#' * round(percent)).ljust(100, '-'),
        rbar_prefix, progress, total, percent)
    sys.stdout.write(buf)
    sys.stdout.write('\r')
    sys.stdout.flush()


def matching():

    print("Loading Words...")
    words = None
    with open(train_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line[0] == '%':
                words = line.strip()[1:].split(',')
                break

    print("Loading Model...")
    model = KeyedVectors.load_word2vec_format('../data/Word2Vec/GoogleNews-vectors-negative300.bin', binary=True)

    w2v = dict()
    non_match = list()

    print("Embedding Matching...")
    total = len(words)
    for i, word in enumerate(words):
        if word in model.vocab:
            vector = model[word]
            w2v[word] = vector
        else:
            non_match.append(word)
        report_progress(i, total)

    with open("../data/Word2Vec/matching.pickle", 'wb') as file:
        pickle.dump(w2v, file)

    with open("../data/Word2Vec/non_match.pickle", 'wb') as file:
        pickle.dump(non_match, file)

    print("Total: [{}], Match: [{}], Non-Match: [{}]".format(len(words), len(w2v), len(non_match)))


def check():
    print("Matchings: ")
    with open("../data/Word2Vec/matching.pickle", 'rb') as file:
        w2v = pickle.load(file)
        print(w2v.items()[:10])

    print("\nNon-Matching: ")
    with open("../data/Word2Vec/non_match.pickle", 'rb') as file:
        non_match = pickle.load(file)
        print(non_match[:10])




if __name__=='__main__':
    train_file = "../data/mxm/mxm_dataset_train.txt"

    matching()
    # check()
