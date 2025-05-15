import math
import torch

# Transformers

import torch.nn as nn

# Embeddings
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, d_model, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=d_model)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)


# Layer norm and feed forward
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # multiplied
        self.beta = nn.Parameter(torch.zeros(d_model)) # added

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2
        self.activation = GELU()

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(self.activation(self.linear_1(x))))

# Attention
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // n_heads
        self.h = n_heads
        assert d_model % n_heads == 0, "d_model is divisible by h"

        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def attention(self, q, k, v, mask=None):
        d_k = q.size(-1)

        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k) # (batch_size, h, seq_len, d_k) @ (batch_size, h, d_k, seq_len) -> (batch_size, h, seq_len, seq_len)

        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9) # if mask is 0, fill with -1e9
        
        attention_scores = attention_scores.softmax(dim=-1) # (batch_size, h, seq_len, seq_len); -1e9 will be close to 0

        
        attention_scores = self.dropout(attention_scores)

        return attention_scores @ v, attention_scores

    def forward(self, q, k, v, mask=None):
        query = self.W_q(q) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        key = self.W_k(k) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        value = self.W_v(v) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)

        # Split the d_model into h heads
        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(1, 2) # (batch_size, seq_len, d_model) -> (batch_size, h, seq_len, d_k)
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1, 2) # (batch_size, seq_len, d_model) -> (batch_size, h, seq_len, d_k)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(1, 2) # (batch_size, seq_len, d_model) -> (batch_size, h, seq_len, d_k)

        # Attention
        x, self.attention_scores = self.attention(query, key, value, mask) 


        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)

        return self.W_o(x) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
    
class ResidualConnection(nn.Module):

    def __init__(self, d_model, eps, dropout):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(d_model, eps)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# Transformer Encoder
class EncoderBlock(nn.Module):

    def __init__(self, d_model, d_ff, n_heads, eps, dropout_enc=0, dropout_ff = 0.1, dropout_rescon= 0.1):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout_enc)
        self.feed_forward = FeedForward(d_model= d_model, d_ff= d_ff, dropout=dropout_ff)
        self.residual_connection = nn.ModuleList([ResidualConnection(d_model=d_model, eps = eps, dropout = dropout_rescon) for _ in range(2)])

    def forward(self, x, src_mask): # src_mask is the mask for the source sequence, no interaction between padding and actual words

        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, src_mask)) # calling the forward method of multiheadattention
        x = self.residual_connection[1](x, self.feed_forward)
        return x 

### Downstream tasks ###
class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))



class BERT(torch.nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, opt):
        super(BERT, self).__init__()

        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """


        self.opt = opt

        if self.opt.device == "mps":
            torch.set_num_threads(8)

        #EMbedding
        self.vocab_size = opt.input.vocab_size
        self.d_model = opt.model.d_model
        self.dropout_embed = opt.model.dropout_embed

        # Encoder
        self.d_ff = opt.model.d_ff
        self.n_heads = opt.model.n_heads
        self.dropout_enc = opt.model.dropout_enc
        self.dropout_ff = opt.model.dropout_ff

        # Residual con and layer norm
        self.eps = opt.model.eps
        self.dropout_rescon = opt.model.dropout_rescon

        self.n_layers = opt.model.n_layers

        # Import functions for BERT LM
        self.next_sentence = NextSentencePrediction(self.d_model)
        self.mask_lm = MaskedLanguageModel(self.d_model, self.vocab_size)

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=self.vocab_size, d_model=self.d_model, dropout=self.dropout_embed)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [EncoderBlock(d_model = self.d_model, d_ff = self.d_ff, n_heads = self.n_heads, eps = self.eps, dropout_enc = self.dropout_enc, dropout_ff = self.dropout_ff, dropout_rescon = self.dropout_rescon) 
                for _ in range(self.n_layers)])

        self.criterion = nn.NLLLoss(ignore_index=0)
        
        # self._init_weights() # If you want to initialize the weights of the model, uncomment this line

        

    def _init_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, "padding_idx") and m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])

        self.embedding.apply(init_fn)
        self.transformer_blocks.apply(init_fn)
        self.next_sentence.apply(init_fn)
        self.mask_lm.apply(init_fn)
        
    
    def forward(self, input, scalar_outputs=None, LM=True):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
                "next_loss": torch.zeros(1, device=self.opt.device),
                "mask_loss": torch.zeros(1, device=self.opt.device),
                "total_correct": torch.zeros(1, device=self.opt.device),
                "total_element": torch.zeros(1, device=self.opt.device),
                "accuracy": torch.zeros(1, device=self.opt.device)
            }
        torch.autograd.set_detect_anomaly(True)

        z = input

        if LM:
            bert_input = z["bert_input"]
            bert_label = z["bert_label"]
            segment_label = z["segment_label"]
            is_next = z["label"]
        
        else:
            bert_input = z["bert_input"]
            segment_label = z["segment_label"]
            is_next = None
            bert_label = None
        
        # print("bert_input", bert_input[0])
        # print("segment_label", segment_label[0])

        # print("bert_input shape:", bert_input.shape)
        # print("segment_label shape:", segment_label.shape)
        # print("bert_label:", bert_label)
        
        mask = (bert_input > 0).unsqueeze(1).repeat(1, bert_input.size(1), 1).unsqueeze(1)
        # print("mask shape:", mask.shape)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(bert_input, segment_label)
        # print("x shape:", x.shape)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        # print("x shape:", x.shape)

        if LM:
            # 1. forward the next_sentence_prediction and masked_lm model
            next_sentence = self.next_sentence(x)
            mask_lm = self.mask_lm(x)

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            # print("next_sentence shape:", next_sentence.shape)
            # print("next_sentence example:", next_sentence[0:5])
            # print("is_next shape:", is_next.shape)
            # print("is_next example:", is_next[0:5])
            
            next_loss = self.criterion(next_sentence, is_next)

            # 2-2. NLLLoss of predicting masked token word
            mask_loss = self.criterion(mask_lm.transpose(1, 2), bert_label)

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            scalar_outputs["next_loss"] += next_loss
            scalar_outputs["mask_loss"] += mask_loss
            scalar_outputs["Loss"] += next_loss + mask_loss

            # next sentence prediction accuracy
            correct = next_sentence.argmax(dim=-1).eq(is_next).sum().item()
            scalar_outputs["total_correct"] += correct
            scalar_outputs["total_element"] += is_next.nelement()
            scalar_outputs["accuracy"] += correct / is_next.nelement()
        
            return scalar_outputs  
        else:
            return x 

        
    

    def predict(self, data, visualize= False):
        
        with torch.no_grad():
            scalar_outputs = self.forward(data)

                
                
        return scalar_outputs



class BERTForClassification(nn.Module):
    def __init__(self, opt):
        """
        :param pretrained_model: Your pretrained BERT model (e.g., the BERT class you already defined).
        :param num_classes: The number of classes for classification.
        """
        super(BERTForClassification, self).__init__()
        
        # Using the pretrained BERT model as the backbone
        self.opt = opt
        self.bert = BERT(opt)
        
        # Classification head: a simple linear layer that maps the output of BERT to the desired number of classes
        self.classifier = nn.Linear(self.bert.d_model, opt.fine_tune.num_classes)
        
        # Dropout layer for regularization (optional but can help prevent overfitting)
        self.dropout = nn.Dropout(0.1)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
    def forward(self, input, scalar_outputs=None):
        # Input: {'bert_input': <token_ids>, 'segment_label': <segment_ids>, 'attention_mask': <mask>}
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
                "accuracy": torch.zeros(1, device=self.opt.device)
            }
        z = input

        bert_input = z["bert_input"]
        segment_label = z["segment_label"]
        label = z["label"]

        # print("label", label[0])
        

        # Pass through the BERT model (feature extraction)
        x = self.bert(input, LM = False)
        
        # Use the output corresponding to the [CLS] token for classification
        cls_output = x[:, 0, :]  # Shape: (batch_size, d_model)
        
        # Apply dropout for regularization
        cls_output = self.dropout(cls_output)
        
        # Pass through the classification layer
        logits = self.classifier(cls_output)  # Shape: (batch_size, num_classes)

        # print("logits shape:", logits.shape)
        # print("is_next shape:", is_next.shape)

        scalar_outputs["Loss"] += self.criterion(logits, label)
        
        # Calculate accuracy
        _, preds = torch.max(logits, dim=1)  # Get the class with the highest probability
        scalar_outputs["accuracy"] += torch.sum(preds == label).float() / label.size(0) if label is not None else None
        
        return scalar_outputs
    
    def predict(self, data, visualize= False):
        with torch.no_grad():
            scalar_outputs = self.forward(data)
                
        return scalar_outputs