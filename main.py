import models
from datasets.dataset import CelebDataset
from datasets.preprocess.embedding import CaptionEmbedder

if __name__=='__main__':
    # Dataset 확인
    mecab_dic = 'C:\mecab\mecab-ko-dic'
    ft_embedder = CaptionEmbedder()
    ft_embedder.fit('captions.csv',mecab_dic)
    train_data = CelebDataset('datasets/debug/captions.csv','datasets/debug/img',embedder=ft_embedder,dic_path=mecab_dic)
    print(train_data[0][1][:5])
    
    # 다음..