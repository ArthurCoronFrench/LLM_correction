from dataset import generate_dataset
from preprocessing import build_vocab
from models import Encoder, Decoder, Seq2Seq

dataset = generate_dataset(20000)

word_to_idx = build_vocab(dataset)

print("Dataset:",len(dataset))
print("Vocab:",len(word_to_idx))