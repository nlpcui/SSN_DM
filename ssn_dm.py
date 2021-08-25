import torch, collections, math
from torch import nn
from torch._C import dtype
from torch.onnx import register_custom_op_symbolic
from transformers import (BertModel, BertTokenizer, AutoConfig)
from utils import *



class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=512):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x



class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.heads = heads
        self.self_attn = MultiHeadAttention(d_model, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter_, inputs, mask):
        if (iter_ != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        context = self.self_attn(input_norm, input_norm, input_norm, self.heads, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList([TransformerEncoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, inputs, mask):
        batch_size, seq_len = inputs.size(0), inputs.size(1)
        pos_emb = self.pos_emb.pe[:, :seq_len]
        # x = states * mask[:, :, None].float()
        inputs = inputs + pos_emb

        for i in range(self.num_inter_layers):
            inputs = self.transformer_inter[i](i, inputs, mask)  # all_sents * max_tokens * dim

        inputs = self.layer_norm(inputs)
        
        return inputs



class SSNDM(torch.nn.Module):
    def __init__(self, args) -> None:
        super(SSNDM, self).__init__()

        self.hidden_size = args.hidden_size
        self.ext_ff_size = args.ext_ff_size
        self.ext_head_num = args.ext_head_num
        self.ext_dropout = args.ext_dropout
        self.ext_layer_num = args.ext_layer_num

        self.sect_num = args.sect_num
        self.max_seg_num = args.max_seg_num
        self.max_seg_len = args.max_seg_len

        self.memory_slots = args.memory_slots
        self.memory_dim = args.memory_dim
        self.memory_hops = args.memory_hops

        self.gat_head_num = args.gat_head_num
        self.per_head_dim = self.hidden_size // self.gat_head_num
        self.gat_dropout = args.gat_dropout

        self.bert_version = args.bert_version

        self.segment_encoder = BertModel.from_pretrained(self.bert_version)
        self.ext_transformer = TransformerEncoder(
            self.hidden_size, self.ext_ff_size, self.ext_head_num, self.ext_dropout, self.ext_layer_num
        )

        self.initial_memory = nn.Parameter(torch.normal(mean=0, std=1, size=(self.memory_slots, self.memory_dim)))

        self.section_embeddings = torch.nn.Embedding(self.sect_num, self.hidden_size)
        self.segment_position_embeddings = torch.nn.Embedding(self.max_seg_num, self.hidden_size)

        self.mlp_pred = torch.nn.Sequential(collections.OrderedDict([
            ('pred_dense_1', torch.nn.Linear(self.hidden_size*3, 1024)),
            ('pred_relu_1', torch.nn.ReLU()),
            ('pred_dense_2', torch.nn.Linear(1024, 512)),
            ('pred_relu_2', torch.nn.ReLU()),
            ('pred_dense_3', torch.nn.Linear(512, 1)),
        ]))

        self.mlp_structural = torch.nn.Sequential(collections.OrderedDict([
            ('structural_dense_1', torch.nn.Linear(self.hidden_size, self.hidden_size)),
            ('structural_tanh_1', torch.nn.Tanh())
        ]))

        self.mlp_gate = torch.nn.Sequential(collections.OrderedDict([
            ('gate_dense_1', torch.nn.Linear(768*2, 1)),
            ('gate_sigmodi_1', torch.nn.Sigmoid()),
        ]))
        self.mlp_merge = torch.nn.Sequential(collections.OrderedDict([
            ('merge_dense_1', torch.nn.Linear(768*2, 768)),
            ('merge_tanh_1', torch.nn.Tanh()),
        ]))

        self.mlp_att_q_lst = nn.ModuleList([torch.nn.Linear(self.hidden_size, self.hidden_size // self.gat_head_num) for i in range(self.gat_head_num)])
        self.mlp_att_k_lst = nn.ModuleList([torch.nn.Linear(self.hidden_size, self.hidden_size // self.gat_head_num) for i in range(self.gat_head_num)])
        self.mlp_att_v_lst = nn.ModuleList([torch.nn.Linear(self.hidden_size, self.hidden_size // self.gat_head_num) for i in range(self.gat_head_num)])



        self.attn_graph = MultiHeadAttention(
            input_dim=self.hidden_size, output_dim=self.hidden_size, dropout=self.gat_dropout
        )

        self.sigmoid = nn.Sigmoid()


    
    def forward(self, batch_size, token_ids, token_types, position_ids, attention_mask, token_section_ids, label_indices, segment_idx, memory, enable_discource=False, enable_memory=True):
        '''
        token_ids: [batch_size, max_segment_len]
        token_types: [batch_size, max_segment_len]
        attention_mask: [batch_size, max_segment_len] # word_attention
        token_section_ids: [batch_size, max_segment_len]
        token_labels: [batch_size, max_segment_len]
        label_indices: [batch_size, max_segment_len]
        label_num: [batch_size, ]
        memory: [batch_size, memory_slots, memory_dim]
        '''

        if memory is None:
            memory = self.initial_memory.unsqueeze(0).repeat(batch_size, 1, 1)
        

        token_ids = torch.reshape(token_ids, (batch_size, -1))
        token_types = torch.reshape(token_types, (batch_size, -1))
        position_ids = torch.reshape(position_ids, (batch_size, -1))
        attention_mask = torch.reshape(attention_mask, (batch_size, -1))
        token_section_ids = torch.reshape(token_section_ids, (batch_size, -1))
        label_indices = torch.reshape(label_indices, (batch_size, -1))
        segment_idx = torch.reshape(segment_idx, (batch_size, -1))

        outputs = self.segment_encoder(
            input_ids = token_ids, 
                attention_mask = attention_mask,
                token_type_ids = token_types,
                position_ids = position_ids,
        )

        hidden_states = outputs['last_hidden_state']
        # sentence_repr = hidden_states[torch.arange(self.batch_size).unsqueeze(1), cls_indices]
        # print('tensor shape: {}'.format*sentence_repr.shape)
        # print('tensor: {}, numel: {}, value_nan_cnt: {}, grad_nan_cnt: {}'.format('hidden_states_bert', hidden_states.numel(), check_nan(hidden_states)[0], check_nan(hidden_states)[1]))
        # pooler_output = outputs['pooler_output']

        hidden_states = self.ext_transformer(hidden_states, mask=label_indices.long())

        # print('hidden state shape: {}'.format(hidden_states.shape))
        # exit(1)

        seg_pos_embeddings = self.segment_position_embeddings(segment_idx)
        hidden_states = hidden_states + seg_pos_embeddings

        if enable_discource:
            section_embeddings = self.section_embeddings(token_section_ids)
            hidden_states = hidden_states + section_embeddings

        hidden_states = self.mlp_structural(hidden_states)
        graph_states = [(hidden_states, memory), ]

        for i in range(self.memory_hops):
            # hidden_states_g = self.multi_head_attention(graph_states[i][0], graph_states[i][1], graph_states[i][1])
            hidden_states_g = self.attn_graph(graph_states[i][0], graph_states[i][1], graph_states[i][1], self.gat_head_num)

            with torch.no_grad():
                memory_g = self.attn_graph(graph_states[i][1], graph_states[i][0], graph_states[i][0], self.gat_head_num, mask=label_indices)

            graph_states.append((hidden_states_g, memory_g))

        hidden_states_g = graph_states[-1][0]
        memory_g = graph_states[-1][1]

        merge_states = torch.cat([
            hidden_states_g + hidden_states, 
            hidden_states_g - hidden_states, 
            hidden_states_g * hidden_states
        ], -1)


        logits_all = self.mlp_pred(merge_states) # (batch_size, seq_len, 1)
        logits_all = torch.reshape(logits_all, (batch_size, -1))

        logits_cls = torch.masked_select(logits_all, label_indices)

        # update memory
        with torch.no_grad():
            logits_all_masked = torch.where(label_indices, logits_all, torch.full_like(logits_all, float('-inf')))
            sent_weight = torch.nn.functional.softmax(logits_all_masked, dim=-1).unsqueeze(-1) # [batch_size, 512, 1]
            sent_weight = torch.transpose(sent_weight, 1, 2) # [batch_size, 1, 512]
            sent_weight_clean = torch.where(torch.isnan(sent_weight), torch.full_like(sent_weight, 0.), sent_weight)
            r_sum = torch.matmul(sent_weight_clean, hidden_states)
            r_sum = r_sum.repeat(1, self.memory_slots, 1)
            update_gate = self.mlp_gate(torch.cat([memory, memory_g], dim=-1))
            memory_ = update_gate * memory + (torch.ones_like(memory_g) - update_gate) * memory_g
            memory_ = self.mlp_merge(torch.cat([memory_, r_sum], dim=-1))

        probs = self.sigmoid(logits_cls)
        return probs, memory_



    def multi_head_attention(self, q, k, v, mask=None):
        # q/k/v: [batch_size, steps, dim]
        outs = []
        for head_idx in range(self.gat_head_num):

            q_ = self.mlp_att_q_lst[head_idx](q)
            k_ = self.mlp_att_k_lst[head_idx](k) # batch_size, k_len, dim
            v_ = self.mlp_att_v_lst[head_idx](v)

            score = torch.matmul(q_, torch.transpose(k_, 1, 2))
            # score = score / torch.sqrt(torch.tensor(float(self.hidden_size)))
            if mask is not None:
                score = torch.where(mask, score, torch.full_like(score, -99999.0))
            score = torch.nn.functional.softmax(score, -1) # batch_size, |q|, |k|
            outs.append(torch.matmul(score, v_))

        out = torch.cat(outs, -1)
        return out
    


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear_key = nn.Linear(input_dim, output_dim)
        self.linear_value = nn.Linear(input_dim, output_dim)
        self.linear_query = nn.Linear(input_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, head_num, mask=None):
        '''
        q/k/v:
        mask: batch_size, q_len, k_len
        '''
        # [batch_size, steps, head_num, per_head_dim]
        batch_size = query.shape[0]

        q_len = query.shape[1]
        k_len = key.shape[1]
        v_len = value.shape[1]

        per_head_dim = self.output_dim // head_num

        query_ = self.linear_query(query)
        key_ = self.linear_key(key)
        value_ = self.linear_value(value)

        query_ = torch.reshape(query_, (batch_size, head_num, q_len, per_head_dim))
        key_ = torch.reshape(key_, (batch_size, head_num, per_head_dim, k_len))
        value_ = torch.reshape(value_, (batch_size, head_num, v_len, per_head_dim))
        
        score = torch.matmul(query_, key_) # batch_size, head_num, q_len, k_len
        if mask is not None: # [batch_size, seq_len]
            mask_score = torch.reshape(mask.float(), (batch_size, 1, 1, -1))
            mask_score = (1 - mask_score.float()) * -99999.0

            score = mask_score + score
        
        score = torch.nn.functional.softmax(score, -1)
        score = self.dropout(score) # strange dropout in Transformer

        outs = torch.matmul(score, value_) # batch_size, head_num, q_len, per_head_dim
        outs = torch.transpose(outs, 1, 2)
        outs = torch.reshape(outs, (batch_size, q_len, head_num*per_head_dim))

        return outs
