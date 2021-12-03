import torch.nn as nn
import torchvision.models as models

class EncoderInception3(nn.Module):
  def __init__(self, output_size, train=False):
    super(EncoderInception3, self).__init__()
    self.train = train
    self.inception = models.inception_v3(pretrained=True, aux_logits=False)
    self.inception.fc = nn.Linear(self.inception.fc.in_features, output_size)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout()

  def forward(self, images):
    features = self.inception(images)

    for name, param in self.inception.named_parameters():
      if 'fc.weight' in name or 'fc.bias' in name:
        param.requires_grad = True
      else :
        param.requires_grad = self.train

    return self.dropout(self.relu(features))