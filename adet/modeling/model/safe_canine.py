import torch
import torch.nn.functional as F
from transformers.models.canine.modeling_canine import CanineModel, CanineEmbeddings

class SafeCanineEmbeddings(CanineEmbeddings):
    def forward(self, input_ids):
        # Original logic up to the pooling part
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # [batch_size, sequence_length, hidden_size]
        char_embeddings = self.char_embeddings(input_ids)
        char_embeddings = char_embeddings.transpose(1, 2)

        if char_embeddings.shape[-1] < 4:
            pooled = F.adaptive_max_pool1d(char_embeddings, output_size=1)
        else:
            pooled = F.max_pool1d(char_embeddings, kernel_size=4, stride=1)

        pooled = pooled.transpose(1, 2)
        return pooled

class SafeCanineModel(CanineModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = SafeCanineEmbeddings(config)
