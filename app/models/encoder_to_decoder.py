# encoder to decoder - 안가영 작품
import torch
import torch.nn as nn

from models.encoder import EncoderInception3
from models.decoder import DecoderLSTM, DecoderAttention

class EncodertoDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, model='lstm', embedder=None):
        super(EncodertoDecoder, self).__init__()
        self.model = model
        if model=='att' and embedder==None:
            assert 'Attention model needs embedder'
            
        self.encoderInception3 = EncoderInception3(embed_size)
        if model=='lstm':
            self.decoderLSTM = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers)
        elif model=='att':
            self.decoderLSTM = DecoderAttention(embedder, embed_size, hidden_size, vocab_size, num_layers)
            
    def forward(self, images, captions):
        features = self.encoderInception3(images)
        outputs = self.decoderLSTM(features, captions)
        return outputs
    
    def caption_image(self, image, embedder, max_length=50):
        result_caption = []
        
        with torch.no_grad():
            x = self.encoderInception3(image.unsqueeze(0)).unsqueeze(1)
            states = None
            
            if self.model=='lstm':
                for _ in range(max_length):
                    hiddens, states = self.decoderLSTM.lstm(x, states)
                    output = self.decoderLSTM.linear(hiddens.unsqueeze(0))
                    predicted = output.argmax(-1)
                    
                    if embedder.i2w[predicted.item()] == "<pad>":
                        break
                    
                    result_caption.append(predicted.item())
                    
                    device = x.device
                    x = embedder.vectorize_caption(predicted.item()).unsqueeze(0).unsqueeze(0)
                    x = x.to(device)
            
                return [embedder.i2w[idx] for idx in result_caption]
            
            elif self.model=='att':
                return [embedder.i2w[idx] for idx in \
                    self.decoderLSTM.greedy_search(x, max_sentence=max_length, stop=embedder.w2i['<pad>'])]