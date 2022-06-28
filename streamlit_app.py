import torch
import pickle5 as p

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = DEVICE

PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX: break
    return ys


def translate(model, src, src_vocab, tgt_vocab, src_tokenizer):
    model.eval()
    
    tokens = [BOS_IDX] + [src_vocab.get_stoi().get(tok,0) for tok in src_tokenizer.encode(src, out_type=str)]+ [EOS_IDX]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1) )
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join([tgt_vocab.get_itos()[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")

def latex_split(eqs):
    syms = []
    buffer = ''
    
    for eq in eqs.split(' '):
        for i in eq:
            buffer += i
            if valid(buffer):
                syms += [buffer]
                buffer = ''
    return syms
            
#DF_SPLITTED = []
#for exp in latexs[:1]:
#    res = latex_split(exp)
#    DF_SPLITTED += [res]
#print(DF_SPLITTED)
#plt.close()

def simple_latex_splitter(t_raw):
    #t_raw = '\\gcd(a,b)=\\prod _{p}p^{\\min(a_{p},b_{p})}'
    res = []
    buffer = ''
    
    command = False
    for char in t_raw:
        if char in ['{','(','[']:
            res += [buffer]
            buffer = ''
            command = True
        if command:
            buffer += char
        else:
            buffer += char
        if char in ['}',')',']']:
            res += [buffer]
            buffer = ''
            command = False
    res += [buffer]
    buffer = ''
    return res


class CustomSplitter:
    def __init__(self, lang):
        self.lang = lang
        
    def encode_rus(self, sentence):
        return sentence.split(' ')
    
    def encode_latex(self, sentence):
        sentence = clear_latex(sentence)
        #sentence = latex_split(sentence)
        sentence = simple_latex_splitter(sentence)
        plt.close()
        plt.close('all')
        return sentence
    
    def encode(self, sentence, out_type):
        if self.lang == 'rus':
            return self.encode_rus(sentence)
        else: return self.encode_latex(sentence)

import math
#import torchtext
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from collections import Counter
#from torchtext.vocab import Vocab
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
import io
import time
import numpy as np
import pickle

from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward:int = 512, dropout:float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

transformer2 = torch.load('model_production.model', map_location=torch.device('cpu'))

with open('ja_vocab_production.pickle', 'rb') as handle:
    ja_vocab2 = p.load(handle)
    
with open('en_vocab_production.pickle', 'rb') as handle:
    en_vocab2 = p.load(handle)
    
with open('tokenizer_production.pickle', 'rb') as handle:
    ja_tokenizer2 = p.load(handle)
         
def tr(word):
    eq = translate(transformer2, word, ja_vocab2, en_vocab2, ja_tokenizer2)
    #display(Latex(f'${eq}$'))
    print(eq)
    return eq

#tests_ = ["игрек плюс икс", 'предел эф от икс при икс стремящемся к альфа',
#         'синус альфа синус бета', 'косинус двух', 'косинус икс', 'логарифм косинуса икс по основанию три','логарифм альфа по основанию пси']
#res = ['x+y', '\lim f(x)', '\sin \alpha', '\cos 2']
##mean_bleu = 0
#for key,test_ in enumerate(tests_):
#    tr(test_)


import streamlit as st

st.header('Descartes: перевод математических формул, записанных словами в LaTeX')
st.markdown('Данная сборка модели обучена на арифметику, включая корни, логарифмы, степени, а также пределы и интегралы из математического анализа, функции суммы и произведения. Модель обучена на примерах минимальной вложенности на русском языке на латинском и греческих алфавитах не включая цифры.')

text = st.text_input('Введите формулу словами', value="два икс плюс три")

def print_ans(string):
    return tr(string)

if st.button('Показать формулу'):
    ans = print_ans(text)
    st.text('Формула в "сыром" LaTeX:')
    st.text(ans)
    
    st.text('Зарендеренная формула:')
    st.latex(ans)

st.text('Баги и идеи на github:')
st.text('Вячеслав Сергеев. sergeev46v@gmail.com')
