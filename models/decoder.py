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
    
class BeamSearch():
    """Class performs the caption generation using Beam search.
    <input>
    - decoder - trained Decoder of captioning model
    - features - feature map outputed from Encoder
    
    <output>
    - sentence - generated caption
    - final_scores - cummulative scores for produced sequences
    """
    def __init__(self, decoder, features, k, max_sentence):
        
        self.k = k
        self.max_sentence = max_sentence
        self.decoder = decoder
        self.features = features
        
        self.h = decoder.init_hidden(features)[0]
        self.c = decoder.init_hidden(features)[1]
        
        self.start_idx = torch.zeros(1).long()
        self.start_score = torch.FloatTensor([0.0]).repeat(k)
        
        # hidden states on the first step for a single word
        self.hiddens = [[[self.h, self.c]]*k]
        self.start_input = [[self.start_idx], self.start_score]
        self.complete_seqs = [list(), list()]
        # track the step
        self.step = 0
        
    def beam_search_step(self):
        """Function performs a single step of beam search, returning start input"""
        top_idx_temp = []
        top_score_temp = []
        hiddens_temp = []
        
        for i, w in enumerate(self.start_input[0][-1]):
            
            hidden_states = self.hiddens[self.step][i]
            h = hidden_states[0]
            c = hidden_states[1]
            # scoring stays with the same dimensions
            embedded_word = self.decoder.embeddings(w.view(-1))
            context, atten_weight = self.decoder.attention(self.features, h)
            # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)
            input_concat = torch.cat([embedded_word, context],  dim = 1)
            
            h, c = self.decoder.lstm(input_concat, (h, c))
            output = self.decoder.fc(h)
            scoring = F.log_softmax(output, dim=1)
            top_scores, top_idx = scoring[0].topk(self.k)
        
            top_cum_score = top_scores + self.start_input[1][i]
            # append top indices and scores
            top_idx_temp.append(top_idx.view(-1, self.k))
            top_score_temp.append(top_cum_score.view(-1, self.k))
            # append hidden states
            hiddens_temp.append([h, c])
        self.hiddens.append(hiddens_temp)
            
        # concatinate temp lists
        top_idx_temp = torch.cat(top_idx_temp, dim =0)
        top_score_temp = torch.cat(top_score_temp, dim =0)
        cum_score = top_score_temp
        
        top_cum_scores = self.get_cummulative_score(cum_score)
        ready_idx, tensor_positions = self.get_ready_idx(top_cum_scores, 
                                                         top_idx_temp,
                                                         cum_score)
        row_pos = self.get_positions(tensor_positions)
        # update the attributes
        self.update_start_input(ready_idx, row_pos, top_cum_scores)
        self.update_hiddens(row_pos)
        self.update_step()
            
        # step == 1 means we have generated the hiddens from <start> word and outputed k first words
        # we use them to generate k second words
        if self.step == 1:
            self.hiddens[self.step] = self.hiddens[self.step] * self.k
            self.start_input[0][0] = self.start_input[0][0].view(self.k,-1)
        
        return  self.start_input
    
    def get_cummulative_score(self, cum_score):
        """Getting the top scores and indices from cum_score"""
        top_cum_scores, _ = cum_score.flatten().topk(self.k)
        return top_cum_scores
    
    def get_ready_idx(self, top_cum_scores, top_idx_temp, cum_score):
        """Obtain a list of ready indices and their positions"""
        # got the list of top positions 
        tensor_positions = [torch.where(cum_score == top_cum_scores[i]) for i in range(self.k)]
        # it's important to sort the tensor_positions by first entries (rows)
        # because rows represent the sequences: 0, 1 or 2 sequences
        tensor_positions = sorted(tensor_positions, key = lambda x: x[0])
        # get read top k indices that will be our input indices for next iteration
        ready_idx = torch.cat([top_idx_temp[tensor_positions[ix]] for ix in range(self.k)]).view(self.k, -1)
        return ready_idx, tensor_positions
        
    def get_positions(self, tensor_positions):
        """Retruns the row positions for tensors"""
        row_pos = [x[0] for x in tensor_positions]
        row_pos = torch.cat(row_pos, dim =0)
        return row_pos
    
    def get_nonend_tokens(self):
        """Get tokens that are not <end>"""
        non_end_token = self.start_input[0][-1] !=1
        return non_end_token.flatten()

    def update_start_input(self, ready_idx, row_pos, top_cum_scores):      
        """Returns new input sequences"""
        # construct new sequence with respect to the row positions
        start_input_new = [x[row_pos] for x in self.start_input[0]]
        self.start_input[0] = start_input_new 
        start_score_new = self.start_input[1][row_pos]
        self.start_input[1] = start_score_new
        
        # append new indices and update scoring
        self.start_input[0].append(ready_idx)
        self.start_input[1] = top_cum_scores.detach()
        
    def update_hiddens(self, row_pos):
        """Returns new hidden states"""
        self.hiddens = [[x[i] for i in row_pos.tolist()] for x in self.hiddens]
        
    def update_step(self):
        """Updates step"""
        self.step += 1
    
    def generate_caption(self):
        """Iterates over the sequences and generates final caption"""
        while True:
            # make a beam search step 
            self.start_input = self.beam_search_step()
        
            non_end_token = self.get_nonend_tokens()
            if (len(non_end_token) != sum(non_end_token).item()) and (sum(non_end_token).item() !=0):
                #prepare complete sequences and scores
                complete_seq = torch.cat(self.start_input[0], dim =1)[non_end_token !=1]
                complete_score = self.start_input[1][non_end_token !=1]
                self.complete_seqs[0].extend(complete_seq)
                self.complete_seqs[1].extend(complete_score)  
            
                start_input_new = torch.cat(self.start_input[0], dim =1)[non_end_token]
                start_input_new = [x.view(len(x), -1) for x in start_input_new.view(len(start_input_new[0]), -1)]
                start_score_new = self.start_input[1][non_end_token]
                
                self.start_input[0] = start_input_new
                self.start_input[1] = start_score_new
                
                non_end_pos = torch.nonzero(non_end_token).flatten()
                self.update_hiddens(non_end_pos)
            elif (sum(non_end_token).item() ==0):
                # prepare complete sequences and scores
                complete_seq = torch.cat(self.start_input[0], dim =1)[non_end_token !=1]
                complete_score = self.start_input[1][non_end_token !=1]
                
                self.complete_seqs[0].extend(complete_seq)
                self.complete_seqs[1].extend(complete_score) 
            else:
                pass
            if (len(self.complete_seqs[0])>=self.k or self.step == self.max_sentence):
                break
        
        return self.get_top_sequence()
            
    def get_top_sequence(self):
        """Gets the sentence and final set of scores"""
        lengths = [len(i) for i in self.complete_seqs[0]]
        final_scores = [self.complete_seqs[1][i] / lengths[i] for i in range(len(lengths))]
        best_score = np.argmax([i.item() for i in final_scores])
        sentence = self.complete_seqs[0][best_score].tolist()
        return sentence, final_scores

# class AttentionLSTM(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers):
#         self.lstm = nn.LSTM(embedding_dim + embedding_dim, hidden_dim, num_layers, batch_first=True)
#         self.attention = BahdanauAttention(embedding_dim, hidden_dim)
#         self.init_h = nn.Linear(embedding_dim, hidden_dim)
#         self.init_c = nn.Linear(embedding_dim, hidden_dim)
        
#     def forward(self, h, c):
        
