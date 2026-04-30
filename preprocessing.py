def build_vocab(dataset):

    vocab = set()

    for wrong, correct in dataset:
        for w in wrong + correct:
            vocab.add(w.lower())

    vocab = list(vocab)
    vocab += ["<PAD>", "<UNK>"]

    return {w:i for i,w in enumerate(vocab)}


def encode(sentence, word_to_idx):

    return [
        word_to_idx.get(w.lower(), word_to_idx["<UNK>"])
        for w in sentence
    ]


MAX_LEN = 12

def pad(seq, pad_idx):

    if len(seq) < MAX_LEN:
        seq += [pad_idx] * (MAX_LEN - len(seq))

    return seq[:MAX_LEN]