import gensim
from gensim.models.keyedvectors import KeyedVectors
import pickle


def matching():
    words = None
    with open(train_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line[0] == '%':
                words = line.strip()[1:].split(',')
                break

    # Load Google's pre-trained Word2Vec model.
    model = KeyedVectors.load_word2vec_format('../data/Word2Vec/GoogleNews-vectors-negative300.bin', binary=True)

    w2v = dict()
    non_match = list()
    for word in words:
        if word in model.vocab:
            vector = gensim_model[word]
            w2v[word] = vector
        else:
            non_match.append(word)

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
