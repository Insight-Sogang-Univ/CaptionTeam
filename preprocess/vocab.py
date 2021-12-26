from konlpy.tag import Mecab
    
class VocabBuilder2():
    def __init__(self, dic_path, how='mecab'):
        self.tokenizer = Mecab(dicpath=dic_path).morphs
        
    def tokenize(self, caption):
        return self.tokenizer(caption)
    
    def tokenize_df(self, df):
        df = df.copy()
        df['tokenized'] = df['caption'].apply(lambda x: ' '.join(self.tokenize(x)))
        return df
    
    def indexize_df(self, df, w2i):
        df = df.copy()
        
        def unk_check(x):
            try:
                return w2i[x]
            except:
                return w2i['<unk>']
            
        df['indexed'] = df['tokenized'].apply(lambda x: ' '.join([f'{unk_check(key)}' for key in x.split()]))
        return df
    
    @staticmethod
    def make_dict(sr, min_count=2):
        # Make dictionary
        dictionary = {}
        for line in sr.values:
            tokens = line.split()
            for token in tokens:
                dictionary[token] = dictionary.get(token, 0) + 1

        # Sort the dictionary
        dictionary = dict(sorted(dictionary.items(), key = lambda x: x[1], reverse=True))                
        
        # Check min_count
        del_keys = []
        for key, item in dictionary.items():
            if item < min_count:
                del_keys.append(key)
        [dictionary.pop(key) for key in del_keys]
        
        # Make the index
        w2i = {'<unk>':0,
               '<pad>':1,
               }
        
        for key, value in dictionary.items():
            w2i[key] = len(w2i)
                
        return w2i