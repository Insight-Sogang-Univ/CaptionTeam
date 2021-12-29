import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        # (BxLxD)
        outputs = self.linear(hiddens)
        # (BxLxV)
        return outputs
   
class DecoderAttention(nn.Module):
    """
    - embedding_dim : size of embeddings
    - hidden_dim : size of LSTM layer (number of hidden states)
    - vocab_size : size of vocabulary
    - p : dropout probability
    """

    def __init__(self, embedder, embedding_dim, hidden_dim, vocab_size, num_layers=1, p=0.5):

        super(DecoderAttention, self).__init__()

        self.num_features = embedding_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.sample_temp = 0.5


        self.embeddings = embedder.vectorize_caption
        # self.lstm = AttentionLSTM(embedding_dim, hidden_dim, vocab_size, num_layers)
        # self.lstm = nn.LSTM(embedding_dim + embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm = nn.LSTMCell(embedding_dim + embedding_dim, hidden_dim)
        self.attention = BahdanauAttention(embedding_dim, hidden_dim)
        self.init_h = nn.Linear(embedding_dim, hidden_dim)
        self.init_c = nn.Linear(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.drop = nn.Dropout(p=p)

    def forward(self, features, captions, sample_prob=0.0):

        """
        <input>
        - captions : image captions (already embeded, BxLxD)
        - features : features returned from Encoder
        - sample_prob : use it for scheduled sampling

        <output>
        - outputs : outputs from t steps
        - attention_weights : weights from attention network
        """

        # create embeddings for captions of size (batch, sqe_len, embed_dim)
        # embed = self.embeddings(captions)
        h, c = self.init_hidden(features)
        seq_len = captions.size(1) + 1 #### 사이즈 조정
        feature_size = features.size(1)
        batch_size = features.size(0)
        # these tensors will store the outputs from lstm cell and attention weights
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size)
        attention_weights = torch.zeros(batch_size, seq_len, feature_size)
        
        # scheduled sampling for training
        for t in range(seq_len):
            if t==0:
                _, attention_weight = self.attention(features, h)
                h = self.drop(h)
                output = self.linear(h)
            else:
                sample_prob = 0.0 if t == 1 else 0.5
                use_sampling = np.random.random() < sample_prob
                if use_sampling == False:
                    word_embed = captions[:, t-1, :]
                context, attention_weight = self.attention(features, h)
                # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)
                input_concat = torch.cat([word_embed, context], 1)
                h, c = self.lstm(input_concat, (h, c))
                h = self.drop(h)
                output = self.linear(h)
                if use_sampling == True:
                    # use sampling temperature to amplify the values before applying softmax
                    scaled_output = output / self.sample_temp
                    scoring = F.log_softmax(scaled_output, dim=1)
                    top_idx = scoring.topk(1)[1]
                    word_embed = torch.stack([self.embeddings(x.item()) for x in top_idx.squeeze()]).squeeze(1)
            outputs[:, t, :] = output
            attention_weights[:, t, :] = attention_weight
        return outputs  #, attention_weights

    def init_hidden(self, features):
        """Initializes hidden state and cell memory using average feature vector.
        <input>
        - features : output from Encoder Bx1xD
        <output>
        - h0 : initial hidden state (short-term memory)
        - c0 : initial cell state (long-term memory)
        """
        # mean_annotations = torch.mean(features, dim=1)
        h0 = features
        c0 = torch.zeros_like(features)
        return h0, c0

    def greedy_search(self, features, stop=None, max_sentence=20):

        """Greedy search to sample top candidate from distribution.
        <input>
        - features : features from Encoder
        - max_sentence : max number of token per caption (default=20)
        <output>
        - sentence : list of tokens
        """
        features = features.squeeze(0)
        sentence = []
        weights = []
        input_word = torch.tensor(0).item()
        h, c = self.init_hidden(features)
        while True:
            embedded_word = self.embeddings(input_word).unsqueeze(0)
            context, attention_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + context size)
            input_concat = torch.cat([embedded_word, context], dim=1)
            h, c = self.lstm(input_concat, (h, c))
            h = self.drop(h)
            output = self.linear(h)
            scoring = F.log_softmax(output, dim=1)
            top_idx = scoring[0].topk(1)[1]
            if top_idx == stop:
                break
            sentence.append(top_idx.item())
            weights.append(attention_weight)
            input_word = top_idx
            if len(sentence) >= max_sentence:
                break
        return sentence #, weights
     
class BahdanauAttention(nn.Module):
    
    def __init__(self, num_features, hidden_dim, output_dim=1):
        super(BahdanauAttention, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # fully-connected layer to learn first weight matrix Wa
        self.W_a = nn.Linear(self.num_features, self.hidden_dim)
        # fully-connected layer to learn the second weight matrix Ua
        self.U_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        # fully-connected layer to produce score (output), learning weight matrix va
        self.v_a = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, features, hidden_size):
        """
        <input>
        - features : output from Encoder
        - hidden_size : hidden state output from Decoder
        <output>
        - context - context vector with a size of (1,2048)
        - attention_weight - probabilities
        """
        hidden_size = hidden_size.unsqueeze(1)
        features = features.unsqueeze(1)
        attention_1 = self.W_a(features)
        attention_2 = self.U_a(hidden_size)
        attention_tan = torch.tanh(attention_1 + attention_2)
        attention_score = self.v_a(attention_tan)
        attention_weight = F.softmax(attention_score, dim=1)
        context = torch.sum(attention_weight * features, dim=1)
        attention_weight = attention_weight.squeeze(dim=2)

        return context, attention_weight
    
def visualize_attention(orig_image, words, atten_weights):
    """Plots attention in the image sample.
  <input>
    - orig_image : image of original size
    - words : list of tokens
    - atten_weights : list of attention weights at each time step 
    """
    fig = plt.figure(figsize=(14,12)) 
    len_tokens = len(words)
    
    for i in range(len(words)):
        atten_current = atten_weights[i].detach().numpy()
        atten_current = atten_current.reshape(7,7)       
        ax = fig.add_subplot(len_tokens//2, len_tokens//2, i+1)
        ax.set_title(words[i])
        img = ax.imshow(np.squeeze(orig_image))
        ax.imshow(atten_current, cmap='gray', alpha=0.8, extent=img.get_extent(), interpolation = 'bicubic')
    plt.tight_layout()
    plt.show()

# class AttentionLSTM(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers):
#         self.lstm = nn.LSTM(embedding_dim + embedding_dim, hidden_dim, num_layers, batch_first=True)
#         self.attention = BahdanauAttention(embedding_dim, hidden_dim)
#         self.init_h = nn.Linear(embedding_dim, hidden_dim)
#         self.init_c = nn.Linear(embedding_dim, hidden_dim)
        
#     def forward(self, h, c):
        
