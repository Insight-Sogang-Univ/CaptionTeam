import torch.nn as nn
import torchvision.models as models

class EncoderInception3(nn.Module):
  def __init__(self, embed_size, Train=False):
    super(EncoderInception3, self).__init__()
    self.Train = Train
    self.inception = models.inception_v3(pretrained=True, aux_logits=False)
    self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout()

  def forward(self, images):
    features = self.inception(images)

    for name, param in self.inception.named_parameters():
      if 'fc.weight' in name or 'fc.bias' in name:
        param.requires_grad = True
      else :
        param.requires_grad = self.Train

    return self.dropout(self.relu(features))