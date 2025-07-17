import torch.nn as nn

class TextToImageTransformer(nn.Module):
    def __init__(self, text_vocab_size, image_vocab_size, n_embd=256, n_head=4, n_layer=4, max_len=256):
        super().__init__()
        self.text_emb = nn.Embedding(text_vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, n_embd))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head), num_layers=n_layer)
        self.lm_head = nn.Linear(n_embd, image_vocab_size)

    def forward(self, text_tokens):
        x = self.text_emb(text_tokens) + self.pos_emb[:, :text_tokens.size(1)]
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits
