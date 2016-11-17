import numpy as np

# This script reads pre-trained embeddings to create an embedding matrix as a numpy array
# and also creates a dictionary that maps a word to its index in that embedding matrix


def get_embedding_matrix_and_word_index_dict(embedding_file_name, vocab_size, embed_size):
    embedding_file = open(embedding_file_name)
    word_index_dict = dict()
    embedding_mtx = np.zeros((vocab_size + 1, embed_size))
    i = 1
    for line in embedding_file:
        line_list = line.split()
        word = line.split()[0]
        word_index_dict[word] = i
        embedding = np.array(line_list[1:])
        embedding_mtx[i, :] = embedding
        i += 1
    return embedding_mtx, word_index_dict

# main function is used for testing
if __name__ == "__main__":
    # embed_file_name = 'glove.6B.50d.txt'
    # embedding_mtx, word_index_dict = get_embedding_matrix_and_word_index_dict(embed_file_name, 400000, 50)
    # print embedding_mtx.shape
    # print embedding_mtx[0]
    # print word_index_dict['war']

    embed_file_name = 'glove.6B.100d.txt'
    embedding_mtx, word_index_dict = get_embedding_matrix_and_word_index_dict(embed_file_name, 400000, 100)
    print embedding_mtx.shape
    print embedding_mtx[0]
    print word_index_dict['war']


