import nltk
import pickle
import os.path
import csv
from pycocotools.coco import COCO
from collections import Counter

class Vocabulary(object):

    def __init__(self,
        vocab_threshold,
        vocab_file='./vocab.pkl',
        start_word="[CLS]",
        end_word="[SEP]",
        unk_word="[UNK]",
        annotations_file='annotations/captions_train2017.json',
        vocab_from_file=False,
        dataset_type = "coco"):
        """Initialize the vocabulary.
        Args:
          vocab_threshold: Minimum word count threshold.
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          annotations_file: Path for train annotation file.
          vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from from existing vocab_file, if it exists
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.dataset_type = dataset_type
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary successfully loaded from vocab.pkl file!')
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)
        
    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def read_captions(self):
        all_captions = {} # combine multi datasets to be implemented
        if(self.dataset_type == 'coco'):
            coco = COCO(self.annotations_file)
            ids = coco.anns.keys()
            captions = {}
            for i, id in enumerate(ids):
                captions[id] = str(coco.anns[id]['caption'])
            return captions
        elif(self.dataset_type == 'vqa'):
            with open('../input/chest-xrays-indiana-university/indiana_reports.csv', mode='r') as report:
                reader = csv.reader(report)
                first_line = next(reader)
                index = first_line.index('findings')  # ['uid', 'MeSH', 'Problems', 'image', 'indication', 'comparison', 'findings', 'impression']
                captions = {rows[0]:rows[index] for rows in reader}
            return captions

    def add_captions(self):
        """Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold."""
        captions = self.read_captions()
        
        counter = Counter()
        ids = captions.keys()
        for i, caption in captions.items():
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
            if ( int(i) %1000 == 0):
                print("[%d/%d] Tokenizing captions..." % (int(i), len(ids)))

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)