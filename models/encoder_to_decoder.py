# encoder to decoder - 안가영 작품
import torch
import torch.nn as nn

from models.encoder import EncoderInception3
from models.decoder import DecoderLSTM

class EncodertoDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(EncodertoDecoder, self).__init__()
        self.encoderInception3 = EncoderInception3(embed_size)
        self.decoderLSTM = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, images, captions):
        features = self.encoderInception3(images)
        outputs = self.decoderLSTM(features, captions)
        return outputs
    
    def caption_image(self, image, vocabulary, embedder, max_length=50):
        result_caption = []
        
        with torch.no_grad():
            print(image.size())
            x = self.encoderInception3(image.unsqueeze(0)).unsqueeze(1)
            states = None
            
            for _ in range(max_length):
                hiddens, states = self.decoderLSTM.lstm(x, states)
                output = self.decoderLSTM.linear(hiddens.unsqueeze(0))
                predicted = output.argmax(-1)
                
                result_caption.append(predicted.item())
                x = embedder.target_transform(predicted).unsqueeze(1)
                
                if vocabulary.itos[predicted.item()] == "<pad>":
                    break
        
        return [vocabulary.itos[idx] for idx in result_caption]