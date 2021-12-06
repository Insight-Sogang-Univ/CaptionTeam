from torchtext.legacy import data
from konlpy.tag import Mecab
from torchtext.legacy.data import TabularDataset

class VocabBuilder():
    def __init__(self, dic_path, fix_length=20, how='mecab'):
        self.fix_length = fix_length
        self.tokenizer = Mecab(dicpath=dic_path)
        self.tokenizer = self.tokenizer.morphs
        self.captionTokens = None
        self.TEXT = data.Field(sequential=True,
                               use_vocab=True,
                               tokenize=self.tokenizer,
                               batch_first=True,
                               fix_length=self.fix_length)
    
    def tokenize(self, caption):
        return self.tokenizer(caption)

    def tokenize_df(self, csv_path, feature='caption', build=True, min_freq=5):
        caption_exs = TabularDataset(
            path=csv_path, format='csv',
            fields=[(feature, self.TEXT)], skip_header=True
            )
        
        self._captionExamples=caption_exs
        self.captionTokens=[example.caption for example in caption_exs]
                        
        if build:
            self.TEXT.build_vocab(caption_exs, min_freq=min_freq)
            return self.captionTokens
        else:
            return self.captionTokens
    
    def build_vocab(self, dataset, min_freq=5):
        self.TEXT.build_vocab(dataset, min_freq)
    