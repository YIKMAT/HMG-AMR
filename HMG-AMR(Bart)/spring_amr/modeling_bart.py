import copy
import math
import random
from typing import *

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from transformers import modeling_bart as bart
from transformers.modeling_utils import BeamHypotheses, calc_banned_ngram_tokens, calc_banned_bad_words_ids, \
    top_k_top_p_filtering
import copy
import numpy as np
from spring_amr.ON_LSTM import ONLSTMStack



# Normal_Head = 13
# MG1_Head = 1
# MG2_Head = 1
# MG3_Head = 1
# MG4_Head = 0
# Layer_begin_index = 0
# Layer_end_index = 1

def extract_backreferences(ids, num_embeddings, backpointer_idx):
    ids_mask = ids >= num_embeddings
    backreferences = ids.clone() - num_embeddings
    backreferences[~ids_mask] = 0
    backreferences += (~ids_mask).long() * torch.arange(
        ids.size(1),
        dtype=ids.dtype,
        device=ids.device)
    ids = ids.clone()
    ids[ids_mask] = backpointer_idx
    return ids, backreferences

# 自己加：针对于Phrase的position Encoder
class SegmentPositionEncoding(nn.Module):
    """
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """
    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            error_msg = "Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(dim)
            raise ValueError(error_msg)
        self.max_len = max_len
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(SegmentPositionEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, position_mask):
        """Embed inputs.

        Args:
            emb : [segment count, segment maxlen, batch, dim]
            position_mask : [segment count, segment maxlen, batch]
        """
        emb = emb * math.sqrt(self.dim)
        assert position_mask is not None
        seq_len = position_mask.sum(dim=0).sum(dim=0)
        assert seq_len.size(0) == emb.size(2)
        if seq_len.max() > self.pe.size(0):
            error_msg = "Sequence length {:d} exceeds max_len {:d}".format(seq_len.max(), self.pe.size(0))
            print(error_msg)
        position_embedding = []
        for b in range(seq_len.size(0)):
            position_embedding.append(self.pe[:seq_len[b]].squeeze(1))
        position_embedding = torch.cat(position_embedding).type_as(emb)
        assert position_embedding.size(0) == position_mask.sum()
        emb[position_mask] = emb[position_mask] + position_embedding
        emb = self.dropout(emb)
        return emb


class AMRBartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: bart.BartConfig, embed_tokens, backpointer_idx):
        super().__init__()

        self.backpointer_idx = backpointer_idx

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens

        if config.static_position_embeddings:
            self.embed_positions = bart.SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = bart.LearnedPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx, #config.extra_pos_embeddings,
            )

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = bart.LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = bart.LayerNorm(config.d_model) if config.normalize_before else None

    #     ------自己加，Hierarchical sentence encoding中对每个层加权的部分
        self.w = nn.Parameter(torch.ones(len(self.layers)))
        self.gamma = nn.Parameter(torch.ones(1))
        self.mg_layer_begin_index = config.mg_layer_begin_index
        self.mg_layer_end_index = config.mg_layer_end_index
        self.hie_mode = config.hie_mode



    #

    def forward(
        self, input_ids, embedded=None, attention_mask=None, mg_dict = None,
    ):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *self.output_hidden_states:* is True.
                - **all_attentions** (List[Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert：Pad的位置是True
        if attention_mask is not None:
            attention_mask = bart.invert_mask(attention_mask)


        input_ids, backreferences = extract_backreferences(
            input_ids, self.embed_tokens.num_embeddings, self.backpointer_idx)
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos

        if embedded is not None:
            x += embedded

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)



        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # all_attentions: 多个 B x T x C x C; each_layer_encoder_outputs: 多个 B x T x C;
        encoder_states, all_attentions = [], []
        each_layer_encoder_outputs = []
        for index, encoder_layer in enumerate(self.layers):
            if self.output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (sinput_ids.dtypeee https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            elif self.mg_layer_begin_index <= index < self.mg_layer_end_index:
                x, attn = encoder_layer(x, attention_mask, mg_dict)
            else:
                x, attn = encoder_layer(x, attention_mask, None)

            if self.output_attentions:
                all_attentions.append(attn)

            each_layer_encoder_outputs.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
        if self.output_hidden_states:
            encoder_states.append(x)

        """
         BEGIN===========================================增加Hierarchical sentence encoding模块==========================================
         输出：mg_sentence_embedding_batch 的shape为 [batch * sentence_num * embedding_dim]
                sentence_mask_batch 的shape为[batch * sentence_num]，补充的为0，未补充为1
        """
        max_sentence_num = 0
        # 循环出一个sample中最多有几个句子
        for each_sentence_len in mg_dict['mg_sentence_len']:
            max_sentence_num = len(each_sentence_len) if len(each_sentence_len) > max_sentence_num else max_sentence_num


        # -----------------------方案1：只取最后一层的单词输出，构建sentence embedding-----------------
        if self.hie_mode == 1:
            mg_sentence_embedding_batch = self.construct_sentence_embedding_by_last_layer(mg_dict['mg_sentence_len'], each_layer_encoder_outputs, x, max_sentence_num)

        # -----------------------方案2：构建encoder每一层的句子表示，再随机选择一层，构建sentence embedding-----------------
        if self.hie_mode == 2:
            mg_sentence_embedding_batch = self.construct_sentence_embedding_by_sample_layer(mg_dict['mg_sentence_len'], each_layer_encoder_outputs, x, max_sentence_num)

        # -----------------------方案3.1：构建encoder每一层的句子表示，用所有layer的表示，构建sentence embedding(对所有层求平均)-----------------
        if self.hie_mode == 3:
            mg_sentence_embedding_batch = self.construct_sentence_embedding_by_all_layer(mg_dict['mg_sentence_len'], each_layer_encoder_outputs, x, max_sentence_num)
            mg_sentence_embedding_batch = torch.mean(mg_sentence_embedding_batch, dim=1)
        # -----------------------方案3.2：构建encoder每一层的句子表示，用所有layer的表示，构建sentence embedding(对所有层自动加权)-----------------
        if self.hie_mode == 4:
            mg_sentence_embedding_batch = self.construct_sentence_embedding_by_all_layer(mg_dict['mg_sentence_len'],
                                                                                         each_layer_encoder_outputs, x,
                                                                                         max_sentence_num)
            w = F.softmax(self.w)
            mg_sentence_embedding_batch_new = torch.zeros_like(mg_sentence_embedding_batch.transpose(0, 1))
            for layer_index, w_each in enumerate(w):
                mg_sentence_embedding_batch_each_layer = self.gamma * (w_each * mg_sentence_embedding_batch.transpose(0, 1)[layer_index])
                mg_sentence_embedding_batch_new[layer_index] = mg_sentence_embedding_batch_each_layer
            mg_sentence_embedding_batch = torch.sum(mg_sentence_embedding_batch_new.transpose(0, 1), dim=1)

        sentence_mask_batch = []
        for mg_sentence_len in mg_dict['mg_sentence_len']:
            mask = np.zeros(max_sentence_num)
            mask[: len(mg_sentence_len)] = 1
            sentence_mask_batch.append(mask)
        sentence_mask_batch = torch.tensor(sentence_mask_batch).to(next(self.parameters()).device)
        """
         END===========================================增加Hierarchical sentence encoding模块==========================================
         """

        # T x B x C -> B x T x C
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)

        return x, encoder_states, all_attentions, mg_sentence_embedding_batch, sentence_mask_batch

    """
        构建句子embedding：方案一：通过对encoder最后一层输出进行mean_pooling。
    """
    def construct_sentence_embedding_by_last_layer(self,mg_sentence_len_list,each_layer_encoder_outputs,x, max_len):
        # 构建一个batch内句子的embedding,后续将数值填入  [batch_num * sentence_num * embedding_dim]
        mg_sentence_embedding_batch = torch.zeros((x.shape[1],max_len,x.shape[2]))
        # 循环，填入每个句子的membedding。
        for index, mg_sentence_len in enumerate(mg_sentence_len_list):
            if len(mg_sentence_len) == 1:
                last_layer_output = each_layer_encoder_outputs[len(each_layer_encoder_outputs)-1].transpose(0, 1)[index]
                last_layer_output_pooling = torch.mean(last_layer_output, dim=0)
                mg_sentence_embedding_batch[index][0] = last_layer_output_pooling
            else:
                current_index = 0
                for sentence_index, each_sentence_len in enumerate(mg_sentence_len):
                    last_layer_output = each_layer_encoder_outputs[len(each_layer_encoder_outputs) - 1].transpose(0, 1)[index][current_index:current_index+each_sentence_len]
                    current_index = current_index + each_sentence_len
                    last_layer_output_pooling = torch.mean(last_layer_output, dim=0)
                    mg_sentence_embedding_batch[index][sentence_index] = last_layer_output_pooling
        return mg_sentence_embedding_batch.to(next(self.parameters()).device)

    """
        构建句子embedding：方案二：通过对encoder每一层输出进行mean_pooling,并batch内的每个样本随机选择一层。
    """
    def construct_sentence_embedding_by_sample_layer(self,mg_sentence_len_list,each_layer_encoder_outputs,x, max_len):
        # 构建一个batch内句子的embedding，后续将数值填入  [batch_num *  sentence_num * embedding_dim]
        mg_sentence_embedding_batch = torch.zeros((x.shape[1], max_len, x.shape[2]))
        for index, mg_sentence_len in enumerate(mg_sentence_len_list):
            random_layer_num = random.randint(1,len(each_layer_encoder_outputs))
            current_index = 0
            for sentence_index, each_sentence_len in enumerate(mg_sentence_len):
                last_layer_output = each_layer_encoder_outputs[random_layer_num].transpose(0, 1)[index][current_index:current_index + each_sentence_len]
                current_index = current_index + each_sentence_len
                last_layer_output_pooling = torch.mean(last_layer_output, dim=0)
                mg_sentence_embedding_batch[index][sentence_index] = last_layer_output_pooling
        # 针对batch内的每一个样本，随机选择一层
        return mg_sentence_embedding_batch.to(next(self.parameters()).device)

    """
        构建句子embedding：方案三：通过对encoder每一层输出进行mean_pooling，获得所有层的表示。
    """
    def construct_sentence_embedding_by_all_layer(self,mg_sentence_len_list,each_layer_encoder_outputs,x, max_len):

        # 构建一个batch内句子的embedding，后续将数值填入  [batch_num * layer_num * sentence_num * embedding_dim]
        mg_sentence_embedding_batch = torch.zeros((x.shape[1], len(each_layer_encoder_outputs), max_len, x.shape[2]))
        for index, mg_sentence_len in enumerate(mg_sentence_len_list):
            for layer_num, each_layer_enc_output in enumerate(each_layer_encoder_outputs):
                current_index = 0
                for sentence_index, each_sentence_len in enumerate(mg_sentence_len):
                    last_layer_output = each_layer_enc_output.transpose(0, 1)[index][current_index:current_index + each_sentence_len]
                    current_index = current_index + each_sentence_len
                    last_layer_output_pooling = torch.mean(last_layer_output, dim=0)
                    mg_sentence_embedding_batch[index][sentence_index][sentence_index] = last_layer_output_pooling
        return mg_sentence_embedding_batch.to(next(self.parameters()).device)



class AMRBartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: bart.BartConfig, embed_tokens: nn.Embedding, backpointer_idx, amr_mode=True):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.backpointer_idx = backpointer_idx

        embed_dim = embed_tokens.embedding_dim

        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = bart.SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = bart.LearnedPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx, #config.extra_pos_embeddings,
            )

        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = bart.LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = bart.LayerNorm(config.d_model) if config.add_final_layer_norm else None

        self.pointer_k = nn.Linear(config.d_model, config.d_model)
        # self.pointer_k.weight.data = self.layers[-1].self_attn.k_proj.weight.data.clone()

        self.pointer_q = nn.Linear(config.d_model, config.d_model)
        # self.pointer_q.weight.data = self.layers[-1].self_attn.q_proj.weight.data.clone()

        # self.pointer_k = nn.Sequential(
        #     nn.Linear(config.d_model, config.decoder_ffn_dim),
        #     nn.GELU(),
        #     nn.Linear(config.decoder_ffn_dim, config.d_model),
        # )
        # self.pointer_q = nn.Sequential(
        #     nn.Linear(config.d_model, config.decoder_ffn_dim),
        #     nn.GELU(),
        #     nn.Linear(config.decoder_ffn_dim, config.d_model),
        # )

        self.amr_mode = amr_mode
        self.hie_layer_begin_index = config.hie_layer_begin_index
        self.hie_layer_end_index = config.hie_layer_end_index

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        sentence_embedding,
        sentence_padding_mask,
        decoder_causal_mask,
        decoder_cached_states=None,
        use_cache=False,
        **unused
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_cached_states (dict or None): dictionary used for storing state during generation.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - hidden states
                - attentions
        """

        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = bart.invert_mask(encoder_padding_mask)

        # check attention mask and invert(sentence_mask)
        if sentence_padding_mask is not None:
            sentence_padding_mask_digit = copy.deepcopy(sentence_padding_mask)
            sentence_padding_mask = bart.invert_mask(sentence_padding_mask)
        # 这里的input_ids是decoder的输入
        input_ids, backreferences = extract_backreferences(
            input_ids,
            self.embed_tokens.num_embeddings,
            self.backpointer_idx)
        # embed positions
        embed_pos = self.embed_positions(input_ids, use_cache=use_cache)
        positions = embed_pos

        # to do this during prediction the old positions should be removed. 取当前一位的
        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)  训练时，output_hidden_states为False;
            if self.output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            # 在测试时，如果decoder_cached_states不为空（记录了之前所有步的所有层），则在此取出每一层的这一步之前的（prev_key、prev_value以及prev_key_padding_mask） 作为 layer_state
            layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None

            if self.hie_layer_begin_index <= idx < self.hie_layer_end_index:
                # layer_past(也就是layer_state)是这一层layer的self_attention、encoder_decoder、以及sentence_encoder_decoder三种attention的记录。（prev_key、prev_value以及prev_key_padding_mask）
                x, layer_self_attn, layer_past = decoder_layer(
                    x,
                    encoder_hidden_states,
                    sentence_embedding=sentence_embedding,
                    sentence_padding_mask=sentence_padding_mask,
                    encoder_attn_mask=encoder_padding_mask,
                    decoder_padding_mask=decoder_padding_mask,
                    layer_state=layer_state,
                    causal_mask=decoder_causal_mask,
                )
            else: #训练时：decoder_causal_mask有值, layer_state为None, decoder_padding_mask有值
                x, layer_self_attn, layer_past = decoder_layer(
                    x,
                    encoder_hidden_states,
                    sentence_embedding=None,
                    sentence_padding_mask=None,
                    encoder_attn_mask=encoder_padding_mask,
                    decoder_padding_mask=decoder_padding_mask,
                    layer_state=layer_state,
                    causal_mask=decoder_causal_mask,
                )

            # 当测试时（use_cache为True），next_decoder_cache记录了decoder所有层的当前时刻及之前时刻的self_attention、encoder_decoder、以及sentence_encoder_decoder三种attention
            # 的prev_key、prev_value以及prev_key_padding_mask。
            if use_cache:
                next_decoder_cache.append(layer_past.copy())
            # all_self_attns是多所有层的attention, (batch_size, head_num, seq_len, seq_len)
            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if self.output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        # 再过一个线性层
        xq = self.pointer_q(x)
        xk = self.pointer_k(x)
        # 预测时，第一步输出的decoder_cached_states为None, 其余的都有值
        if decoder_cached_states is not None:
            if 'prev_key' in decoder_cached_states[-1].get('pointer', {}):
                last_state = decoder_cached_states[-1]['pointer']
                xk = torch.cat([last_state['prev_key'], xk], dim=1)

        next_state = {'pointer': {'prev_key': xk}}
        # next_decoder_cache记录了当前这一步以及之前的每一层的self_attention、encoder_decoder、以及sentence_encoder_decoder。（prev_key、prev_value以及prev_key_padding_mask）。
        if use_cache:
            next_decoder_cache.append(next_state)

        if self.amr_mode:
            scores = torch.einsum('bqh,bkh->bqk', xq, xk)

            if decoder_cached_states:
                mask = torch.full_like(scores[0], float('-inf'))
                mask = mask.triu(diagonal=xk.size(1) - 1)
            else:
                mask = torch.full_like(scores[0], float('-inf'))
                mask = mask.triu()
            scores += mask.unsqueeze(0)
        else:
            scores = torch.full((xq.size(0), xq.size(1), xk.size(1)), float('-inf'), device=xq.device)
        # -----在预测时，将上一步的decoder输出记录下来，用于下一步的输入（key）
        # 这里增加句子的embedding 和 句子的mask. sentence_embedding, sentence_padding_mask。在这里，encoder_padding_mask和sentence_padding_mask都是True 或者 False
        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask, sentence_embedding, sentence_padding_mask_digit), next_decoder_cache)
        else:
            next_cache = None
        #    （decoder输出，以及分数），当前及之前所有时刻的所有层的三种attention的k/v记录，所有额隐层状态，所有的attention分布
        return (x, scores), next_cache, all_hidden_states, list(all_self_attns)


class AMRBartModel(bart.PretrainedBartModel):
    def __init__(self, config: bart.BartConfig, backpointer_idx=None):
        super().__init__(config)
        self.output_attentions = True
        self.output_hidden_states = config.output_hidden_states

        self.padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, self.padding_idx)

        if backpointer_idx is not None:
            self.backpointer_idx = backpointer_idx
        else:
            self.backpointer_idx = self.shared.num_embeddings - 1

        self.encoder = AMRBartEncoder(config, self.shared, backpointer_idx=self.backpointer_idx)
        self.decoder = AMRBartDecoder(config, self.shared, backpointer_idx=self.backpointer_idx)

        self.init_weights()

    @property
    def sentence_mode(self):
        return self.decoder.amr_mode

    @sentence_mode.setter
    def sentence_mode(self, value):
        assert isinstance(value, bool)
        self.decoder.amr_mode = value

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        use_cache=False,
        mg_dict=None,
    ):

        # make masks if user doesn't supply。如果是训练阶段，这生成decoder的inputs、padding以及causal_mask。如果是预测阶段，则decoder_input_ids是传进来的，decoder_padding_mask, causal_mask为None
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = bart._prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, mg_dict=mg_dict)
        assert isinstance(encoder_outputs, tuple)
        # encoder_outputs 包括 (x, encoder_states, all_attentions, mg_sentence_embedding_batch, sentence_mask_batch)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # encoder_outputs[-2]是sentence_embedding, encoder_outputs[-1]是sentence_mask。
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_padding_mask,
            encoder_outputs[-2],
            encoder_outputs[-1],
            decoder_causal_mask=causal_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        # Attention and hidden_states will be [] or None if they aren't needed
        # decoder_outputs: Tuple = bart._filter_out_falsey_values(decoder_outputs)
        assert isinstance(decoder_outputs[0][0], torch.Tensor)
        assert isinstance(decoder_outputs[0][1], torch.Tensor)
        encoder_outputs: Tuple = bart._filter_out_falsey_values(encoder_outputs)
        return decoder_outputs + encoder_outputs

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return bart._make_linear_from_emb(self.shared)  # make it on the fly


class AMRBartForConditionalGeneration(bart.PretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, config: bart.BartConfig, backpointer_idx=None):
        super().__init__(config)
        base_model = AMRBartModel(config, backpointer_idx)
        self.model = base_model
        self.pad_index = base_model.shared.padding_idx
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.backpointer_idx = backpointer_idx
        self._rev = None

    def init_reverse_model(self):
        rev = AMRBartForConditionalGeneration(self.model.config, self.backpointer_idx)
        rev.model.shared = self.model.shared
        rev.model.encoder = self.model.encoder
        rev.model.decoder.embed_tokens = self.model.decoder.embed_tokens
        rev.model.decoder.embed_positions = self.model.decoder.embed_positions
        self.amr_mode = True
        rev.amr_mode = False
        self._rev = rev

    @property
    def rev(self):
        if self._rev is None:
            return self
        else:
            return self._rev

    @property
    def amr_mode(self):
        return self.model.decoder.amr_mode

    @amr_mode.setter
    def amr_mode(self, value):
        assert isinstance(value, bool)
        self.model.decoder.amr_mode = value

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        lm_labels=None,
        use_cache=False,
        # phrase_batch=None,
        # phrase_len_batch=None,
        **unused
    ):
        r"""
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels
            in ``[0, ..., config.vocab_size]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

            # Mask filling only works for bart-large
            from transformers import BartTokenizer, BartForConditionalGeneration
            tokenizer = BartTokenizer.from_pretrained('bart-large')
            TXT = "My friends are <mask> but they eat too many carbs."
            model = BartForConditionalGeneration.from_pretrained('bart-large')
            input_ids = tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']
            logits = model(input_ids)[0]
            masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            values, predictions = probs.topk(5)
            tokenizer.decode(predictions).split()
            # ['good', 'great', 'all', 'really', 'very']
        """
        # outputs = self.model(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     decoder_input_ids=decoder_input_ids,
        #     encoder_outputs=encoder_outputs,
        #     decoder_attention_mask=decoder_attention_mask,
        #     decoder_cached_states=decoder_cached_states,
        #     use_cache=use_cache,
        # )
        # lm_logits = F.linear(outputs[0][0], self.model.shared.weight, bias=self.final_logits_bias)
        # po_logits = outputs[0][1]
        # po_padding = torch.full_like(po_logits[:, :, 0:1], float('-inf'))
        # po_padding = po_padding.repeat(1, 1, 1024 - po_logits.size(-1))
        # po_logits = torch.cat([po_logits, po_padding], -1)
        # uni_logits = torch.cat([lm_logits, po_logits], -1)
        #
        # outputs = (uni_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here

        outputs = self.compute_logits(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            mg_dict = unused
        )
        # lm_labels是decoder的输出，用于计算loss
        if lm_labels is not None:
            uni_logits = outputs[0]
            masked_lm_loss = F.nll_loss(
                uni_logits.log_softmax(-1).contiguous().view(-1, uni_logits.size(-1)),
                lm_labels.contiguous().view(-1),
                ignore_index=self.pad_index)
            outputs = (masked_lm_loss,) + outputs

        return outputs

    def compute_logits(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        use_cache=False,
        mg_dict = None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            mg_dict=mg_dict
        )

        lm_logits = F.linear(outputs[0][0], self.model.shared.weight, bias=self.final_logits_bias)
        po_logits = outputs[0][1]
        po_padding = torch.full_like(po_logits[:, :, 0:1], float('-inf'))
        po_padding = po_padding.repeat(1, 1, 1024 - po_logits.size(-1))
        po_logits = torch.cat([po_logits, po_padding], -1)
        uni_logits = torch.cat([lm_logits, po_logits], -1)
        outputs = (uni_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
        return outputs

    @torch.no_grad()
    def generate(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None,
            do_sample: Optional[bool] = None,
            early_stopping: Optional[bool] = None,
            num_beams: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            bad_words_ids: Optional[Iterable[int]] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_start_token_id: Optional[int] = None,
            # phrase_batch: Optional[torch.LongTensor] = None,
            # phrase_len_batch: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            **model_specific_kwargs
    ) -> torch.LongTensor:
        r""" Generates sequences for models with a LM head. The method currently supports greedy decoding, beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

        Adapted in part from `Facebook's XLM beam search code`_.

        .. _`Facebook's XLM beam search code`:
           https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529


        Parameters:

            input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `torch.LongTensor` of shape `(1,)`.

            max_length: (`optional`) int
                The max length of the sequence to be generated.  Between `min_length` and infinity. Default to 20.

            min_length: (`optional`) int
                The min length of the sequence to be generated.  Between 0 and infinity. Default to 0.

            do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is used. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

            early_stopping: (`optional`) bool
                if set to `True` beam search is stopped when at least `num_beams` sentences finished per batch. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

            num_beams: (`optional`) int
                Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.

            temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.

            top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.

            top_p: (`optional`) float
                The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

            repetition_penalty: (`optional`) float
                The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.

            pad_token_id: (`optional`) int
                Padding token. Default to specicic model pad_token_id or None if it does not exist.

            bos_token_id: (`optional`) int
                BOS token. Defaults to `bos_token_id` as defined in the models config.

            eos_token_id: (`optional`) int
                EOS token. Defaults to `eos_token_id` as defined in the models config.

            length_penalty: (`optional`) float
                Exponential penalty to the length. Default to 1.

            no_repeat_ngram_size: (`optional`) int
                If set to int > 0, all ngrams of size `no_repeat_ngram_size` can only occur once.
            bad_words_ids: (`optional`) list of lists of int
                `bad_words_ids` contains tokens that are not allowed to be generated. In order to get the tokens of the words that should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.

            num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each element in the batch. Default to 1.

            attention_mask (`optional`) obj: `torch.LongTensor` of same shape as `input_ids`
                Mask to avoid performing attention on padding token indices.
                Mask values selected in ``[0, 1]``:
                ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
                Defaults to `None`.

                `What are attention masks? <../glossary.html#attention-mask>`__

            decoder_start_token_id=None: (`optional`) int
                If an encoder-decoder model starts decoding with a different token than BOS.
                Defaults to `None` and is changed to `BOS` later.

            use_cache: (`optional`) bool
                If `use_cache` is True, past key values are used to speed up decoding if applicable to model. Defaults to `True`.

            model_specific_kwargs: (`optional`) dict
                Additional model specific kwargs will be forwarded to the `forward` function of the model.

        Return:

            output: `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`
                sequence_length is either equal to max_length or shorter if all batches finished early due to the `eos_token_id`

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            outputs = model.generate(max_length=40)  # do greedy decoding
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3)  # 3 generate sequences using by sampling
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
            input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
        """

        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
                isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
                isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
                isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
                isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
                isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
                bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                        num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                        num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
            pad_token_id = eos_token_id

        # current position and vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
                self.config.is_encoder_decoder
                and hasattr(self.config, "decoder")
                and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size

        vocab_size += 1024

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id

            assert (
                    decoder_start_token_id is not None
            ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
            assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

            # get encoder and store encoder outputs
            encoder = self.get_encoder()
            # encoder_outputs：{x, encoder_states, all_attentions, mg_sentence_embedding_batch, sentence_mask_batch}
            encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask, mg_dict = model_specific_kwargs)
        #     ===自己加：句子的mask
            attention_mask_sentence = encoder_outputs[-1]

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )
            #     ===自己加：句子的mask
            input_sentence_len = attention_mask_sentence.shape[1]
            attention_mask_sentence = attention_mask_sentence.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_sentence_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            #     ===自己加：句子的mask
            attention_mask_sentence = attention_mask_sentence.contiguous().view(
                effective_batch_size * num_beams, input_sentence_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            assert (
                    batch_size == encoder_outputs[0].shape[0]
            ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                    .view(-1, 1)
                    .repeat(1, num_beams * effective_batch_mult)
                    .view(-1)
                    .to(input_ids.device)
            )
            # expand encoder_outputs
            encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])
            # =========扩展sentence_embedding, 自己加. 用于测试时的beam_size大于5==========
            sentence_embedding = encoder_outputs[-2].index_select(0, expanded_batch_idxs)
            encoder_outputs = list(encoder_outputs)
            encoder_outputs[-2] = sentence_embedding
            encoder_outputs[-1] = attention_mask_sentence
            encoder_outputs = tuple(encoder_outputs)


        else:
            encoder_outputs = None
            cur_len = input_ids.shape[-1]

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )

        return output

    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        bos_token_id,
        pad_token_id,
        eos_token_id,
        decoder_start_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        model_specific_kwargs,
    ):
        """ Generate sequences for each example with beam search.
        """

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if do_sample is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
            )
            outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs[1]

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                self.enforce_repetition_penalty_(
                    next_token_logits, batch_size, num_beams, input_ids, repetition_penalty,
                )

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if self.config.is_encoder_decoder and do_sample is False:
                # TODO (PVP) still a bit hacky here - there might be a better solution
                next_token_logits = self.prepare_logits_for_generation(
                    next_token_logits, cur_len=cur_len, max_length=max_length
                )

            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            # set eos token prob to zero if min_length is not reached
            if eos_token_id is not None and cur_len < min_length:
                scores[:, eos_token_id] = -float("inf")

            if no_repeat_ngram_size > 0:
                # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                num_batch_hypotheses = batch_size * num_beams
                # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                banned_batch_tokens = calc_banned_ngram_tokens(
                    input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
                )
                for i, banned_tokens in enumerate(banned_batch_tokens):
                    scores[i, banned_tokens] = -float("inf")

            if bad_words_ids is not None:
                # calculate a list of banned tokens according to bad words
                banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                for i, banned_tokens in enumerate(banned_tokens):
                    scores[i, banned_tokens] = -float("inf")

            assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
                scores.shape, (batch_size * num_beams, vocab_size)
            )

            if do_sample:
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # Top-p/top-k filtering
                _scores = top_k_top_p_filtering(
                    _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                _scores = _scores.contiguous().view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

            else:
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                        eos_token_id is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence or last iteration
                    if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                    else:
                        # add next predicted token if it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if were done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len=cur_len
                )

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1)

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch and update current length
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1

            # re-order internal states
            if past is not None:
                past = self._reorder_cache(past, beam_idx)

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        # finalize all open beam hypotheses and end to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if eos_token_id is not None and all(
                (token_id % vocab_size).item() is not eos_token_id for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
                ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

        # select the best hypotheses
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)

        # shorter batches are filled with pad_token
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_id
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

        return decoded

    @staticmethod
    def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
        return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)
    # 将原Bart的词汇表替换成新的词汇表
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
    # 在这里组装decoder的输入。由于已经有了encoder_inputs，所以input_ids不太需要了
    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"
        # past ： ((encoder_hidden_states, encoder_padding_mask, sentence_embedding, sentence_padding_mask), next_decoder_cache)
        # first step, decoder_cached_states are empty
        if not past[1]:
            encoder_outputs, decoder_cached_states = past, None
        else:
            encoder_outputs, decoder_cached_states = past
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_logits_for_generation(self, logits, cur_len, max_length):
        #if cur_len == 1:
        #    self._force_token_ids_generation(logits, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.config.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        # ((enc_out, enc_mask), decoder_cached_states) = past
        # 自己加
        ((enc_out, enc_mask, sen_enc_out, sen_enc_mask), decoder_cached_states) = past
        reordered_past = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: bart._reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = enc_out if enc_out is None else enc_out.index_select(0, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)
        # 自己加
        new_sen_enc_out = sen_enc_out if sen_enc_out is None else sen_enc_out.index_select(0, beam_idx)
        new_sen_enc_mask = sen_enc_mask if sen_enc_mask is None else sen_enc_mask.index_select(0, beam_idx)

        # past = ((new_enc_out, new_enc_mask), reordered_past)
        # 自己加
        past = ((new_enc_out, new_enc_mask, new_sen_enc_out, new_sen_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return bart._make_linear_from_emb(self.model.shared)  # make it on the fly


# -----------------------------自己加：-----------------------
class DecoderLayer(nn.Module):
    def __init__(self, config: bart.BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.self_attn = SelfAttention(
            embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = bart.ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = bart.LayerNorm(self.embed_dim)
        self.encoder_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_sen_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
            sentence_encoder_decoder_attention=True
        )

        self.encoder_attn_layer_norm = bart.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = bart.LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_hidden_states,
        sentence_embedding=None,
        sentence_padding_mask=None,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
    ):
        residual = x

        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)



        # Self Attention===================
        #  测试时，layer_state是当前时刻之前的所有时刻的所有层的k/v. key_padding_mask为None
        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            need_weights=self.output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention================
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        # 测试时，layer_state是当前时刻之前的所有时刻的所有层的k/v. key_padding_mask为encoder_attn_mask
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Hierarchical sentence attention==============
        if sentence_embedding!=None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            # 测试时，layer_state是当前时刻之前的所有时刻的所有层的k/v. key_padding_mask为sentence_padding_mask
            x, _ = self.encoder_sen_attn(
                query=x,
                key=sentence_embedding,
                key_padding_mask=sentence_padding_mask,
                layer_state=layer_state,  # mutates layer state
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + 0.1 * x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)



        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding



class EncoderLayer(nn.Module):
    def __init__(self, config: bart.BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.self_attn = SelfAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout,
        )
        self.self_attn_mg = SelfAttentionMG(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, head_list=config.head_list
        )
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = bart.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = bart.ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = bart.LayerNorm(self.embed_dim)
        self.head_list = config.head_list
        self.lstm_state = config.lstm_state
        self.lstm = nn.LSTM(self.embed_dim,self.embed_dim, num_layers=2)
        self.on_lstm =  ONLSTMStack([self.embed_dim,self.embed_dim], chunk_size=8)

    def forward(self, x, encoder_padding_mask, mg_dict):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if mg_dict != None:
            head_list = [int(num) for num in self.head_list.split("-")]
            MG_all_list = self.get_multi_granularity(x, mg_dict, head_list)
            # =================这里调用LSTM==============
            if self.lstm_state == 1:
                if head_list[0]!=0:
                    p0 = MG_all_list[0]["key"]
                    p0_lstm_out, (_, _) = self.lstm(p0)
                    MG_all_list[0]["key"] = p0_lstm_out
                    MG_all_list[0]["value"] = p0_lstm_out
                if head_list[1]!=0:
                    p1 = MG_all_list[1]["key"]
                    p1_lstm_out, (_, _) = self.lstm(p1)
                    MG_all_list[1]["key"] = p1_lstm_out
                    MG_all_list[1]["value"] = p1_lstm_out
                if head_list[2]!=0:
                    p2 = MG_all_list[2]["key"]
                    p2_lstm_out, (_, _) = self.lstm(p2)
                    MG_all_list[2]["key"] = p2_lstm_out
                    MG_all_list[2]["value"] = p2_lstm_out
                if head_list[3]!=0:
                    p3 = MG_all_list[3]["key"]
                    p3_lstm_out, (_, _) = self.lstm(p3)
                    MG_all_list[3]["key"] = p3_lstm_out
                    MG_all_list[3]["value"] = p3_lstm_out
            # =================这里调用ON-LSTM=================
            if self.lstm_state == 2:
                if head_list[0]!=0:
                    p0 = MG_all_list[0]["key"]
                    p0_lstm_out = self.on_lstm(p0, self.on_lstm.init_hidden(p0.shape[1]))
                    MG_all_list[0]["key"] = p0_lstm_out[0]
                    MG_all_list[0]["value"] = p0_lstm_out[0]
                if head_list[1]!=0:
                    p1 = MG_all_list[1]["key"]
                    p1_lstm_out = self.on_lstm(p1, self.on_lstm.init_hidden(p1.shape[1]))
                    MG_all_list[1]["key"] = p1_lstm_out[0]
                    MG_all_list[1]["value"] = p1_lstm_out[0]
                if head_list[2]!=0:
                    p2 = MG_all_list[2]["key"]
                    p2_lstm_out = self.on_lstm(p2, self.on_lstm.init_hidden(p2.shape[1]))
                    MG_all_list[2]["key"] = p2_lstm_out[0]
                    MG_all_list[2]["value"] = p2_lstm_out[0]
                if head_list[3]!=0:
                    p3 = MG_all_list[3]["key"]
                    p3_lstm_out = self.on_lstm(p3, self.on_lstm.init_hidden(p3.shape[1]))
                    MG_all_list[3]["key"] = p3_lstm_out[0]
                    MG_all_list[3]["value"] = p3_lstm_out[0]

            x, attn_weights = self.self_attn_mg(
                query=x, key=x, key_padding_mask=encoder_padding_mask, need_weights=self.output_attentions, MG_all_list=MG_all_list
            )
        else:
            x, attn_weights = self.self_attn(
                query=x, key=x, key_padding_mask=encoder_padding_mask, need_weights=self.output_attentions
            )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn_weights


    # 获取不同粒度的表示（key,value等）
    # key value 维度：`(seq_len, batch, embed_dim)`
    # padding_mask : pad为True
    def get_multi_granularity(self, x, mg_dict, head_list):
        emb = x.transpose(0, 1)
        MG_all_list = []
        # HEAD_NUM_LIST = [MG1_Head, MG2_Head, MG3_Head, MG4_Head]
        # ----------------------------------------获取Multi-Phrase的表示---------------------------------
        for i, key in enumerate(mg_dict):
            # -----------去掉'mg_sentence_len'----------
            if key == "mg_sentence_len":
                continue
            mg_phrase_list = []
            mg_phrase_seqLen_list = []
            for sample_index, mg_token in enumerate(emb):
                current_index = 0
                current_emb = torch.zeros(1, x.shape[2])
                for mg_index, mg_len in enumerate(mg_dict[key][sample_index]):
                    emb_new = emb[sample_index][current_index:current_index + mg_len].sum(dim=0) / (mg_len + 1e-10)
                    emb_new = emb_new.unsqueeze(0)
                    current_index = current_index + mg_len
                    if mg_index == 0:
                        current_emb = emb_new
                    else:
                        current_emb = torch.cat((current_emb, emb_new))
                mg_phrase_list.append(current_emb)
                mg_phrase_seqLen_list.append(current_emb.shape[0])
            mg_phrase_batch = torch.nn.utils.rnn.pad_sequence(mg_phrase_list)
            mg_phrase_batch_padding_mask = self.sequence_mask(torch.tensor(mg_phrase_seqLen_list))
            mg_phrase_dict = {}
            mg_phrase_dict["key"] = mg_phrase_batch
            mg_phrase_dict["value"] = mg_phrase_batch
            mg_phrase_dict["padding_mask"] = bart.invert_mask(mg_phrase_batch_padding_mask)
            mg_phrase_dict["head_num"] = head_list[i]
            MG_all_list.append(mg_phrase_dict)

        # -----------------------------------------获取Phrase 2---------------------------------
        # mg_phrase_1_list = []
        # mg_phrase_1_seqLen_list = []
        # for sample_index, mg_token in enumerate(emb):
        #     current_index = 0
        #     current_emb = torch.zeros(1, x.shape[2])
        #     for mg_index, mg_len in enumerate(mg_dict['mg_phrase_1_len'][sample_index]):
        #         emb_new = emb[sample_index][current_index:current_index + mg_len].sum(dim=0) / (mg_len + 1e-10)
        #         emb_new = emb_new.unsqueeze(0)
        #         current_index = current_index + mg_len
        #         if mg_index == 0:
        #             current_emb = emb_new
        #         else:
        #             current_emb = torch.cat((current_emb, emb_new))
        #     mg_phrase_1_list.append(current_emb)
        #     mg_phrase_1_seqLen_list.append(current_emb.shape[0])
        # mg_phrase_1_batch = torch.nn.utils.rnn.pad_sequence(mg_phrase_1_list)
        # mg_phrase_1_batch_padding_mask = self.sequence_mask(torch.tensor(mg_phrase_1_seqLen_list))
        # phrase_1_dict = {}
        # phrase_1_dict["key"] = mg_phrase_1_batch
        # phrase_1_dict["value"] = mg_phrase_1_batch
        # phrase_1_dict["padding_mask"] = bart.invert_mask(mg_phrase_1_batch_padding_mask)
        # phrase_1_dict["head_num"] = 1
        # MG_all_list.append(phrase_1_dict)


        return MG_all_list


    # 生成mask
    def sequence_mask(self, lengths, max_len=None):
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return (torch.arange(0, max_len, device=lengths.device)
                .type_as(lengths)
                .unsqueeze(0).expand(batch_size, max_len)
                .lt(lengths.unsqueeze(1)))





class SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
        sentence_encoder_decoder_attention=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"
        # ========自己加
        self.sentence_encoder_decoder_attention = sentence_encoder_decoder_attention
        if self.sentence_encoder_decoder_attention:
            self.cache_key = "sentence_encoder_decoder"

    def _shape(self, tensor, dim_0, bsz):
        return tensor.contiguous().view(dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)
    # 在预测时，self-attention时 key_padding_mask是None，而cross-attention时，是encoder的padding_mask.
    # layer_state是当前时刻之前的所有时刻的所有层的k/v
    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        # static_kv为True, k和v不用计算，可以使用静态的。
        static_kv: bool = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute key and value if they are static
                if static_kv:
                    key = None
        else:
            saved_state = None
            layer_state = {}
        # ------------这里可以理解成normal-head计算部分--------
        q = self.q_proj(query) * self.scaling
        if static_kv:
            # cross-attention
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            #  self-attention
            k = self.k_proj(query)
            v = self.v_proj(query)
        # 此处变为了[batch_size*num_heads，查询的个数，num_hiddens/num_heads]
        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)
        # 在测试阶段，如果是self-attention（也就是static_kv是False），则拼接之前的k和此次的k（v同理），key_padding_mask是None，因为是按顺序生成的，没必要用这个；
        # 如果是encoder-decoder（也就是static_kv是True），则使用之前的k/v。key_padding_mask是None，因为是调用时都会传入
        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache。 测试阶段，将上一个函数_use_saved_state已经合并的当前步骤及之前所有步的key.value，存起来。
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training,)

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
        key_padding_mask = self._cat_prev_key_padding_mask(
            key_padding_mask, prev_key_padding_mask, bsz, k.size(1), static_kv
        )
        return k, v, key_padding_mask

    @staticmethod
    def _cat_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)

        elif key_padding_mask is not None:
            filler = torch.zeros(
                batch_size,
                src_len - key_padding_mask.size(1),
                dtype=key_padding_mask.dtype,
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask



class SelfAttentionMG(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
            head_list =None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.head_list = [int(num) for num in head_list.split("-")]
        self.normal_heads = self.head_list[4]
        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj_normal = nn.Linear(embed_dim, int(embed_dim/num_heads * self.normal_heads), bias=bias)
        self.v_proj_normal = nn.Linear(embed_dim, int(embed_dim/num_heads * self.normal_heads), bias=bias)
        self.q_proj_normal = nn.Linear(embed_dim, int(embed_dim/num_heads * self.normal_heads), bias=bias)
        # -----------------Phrase 1
        if self.head_list[0]!=0:
            self.k_proj_mg1 = nn.Linear(embed_dim, int(embed_dim/num_heads * self.head_list[0]), bias=bias)
            self.v_proj_mg1 = nn.Linear(embed_dim, int(embed_dim/num_heads * self.head_list[0]), bias=bias)
            self.q_proj_mg1 = nn.Linear(embed_dim, int(embed_dim/num_heads * self.head_list[0]), bias=bias)
        # -----------------Phrase 2
        if self.head_list[1]!=0:
            self.k_proj_mg2 = nn.Linear(embed_dim, int(embed_dim/num_heads * self.head_list[1]), bias=bias)
            self.v_proj_mg2 = nn.Linear(embed_dim, int(embed_dim/num_heads * self.head_list[1]), bias=bias)
            self.q_proj_mg2 = nn.Linear(embed_dim, int(embed_dim/num_heads * self.head_list[1]), bias=bias)
        # -----------------phrase 3
        if self.head_list[2]!=0:
            self.k_proj_mg3 = nn.Linear(embed_dim, int(embed_dim/num_heads * self.head_list[2]), bias=bias)
            self.v_proj_mg3 = nn.Linear(embed_dim, int(embed_dim/num_heads * self.head_list[2]), bias=bias)
            self.q_proj_mg3 = nn.Linear(embed_dim, int(embed_dim/num_heads * self.head_list[2]), bias=bias)
        #  ============clause 4
        if self.head_list[3]!=0:
            self.k_proj_mg4 = nn.Linear(embed_dim, int(embed_dim/num_heads * self.head_list[3]), bias=bias)
            self.v_proj_mg4 = nn.Linear(embed_dim, int(embed_dim/num_heads * self.head_list[3]), bias=bias)
            self.q_proj_mg4 = nn.Linear(embed_dim, int(embed_dim/num_heads * self.head_list[3]), bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor, dim_0, bsz, heads):
        return tensor.contiguous().view(dim_0, bsz * heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights=False,
            MG_all_list = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        # ------------------------------这里可以理解成normal-head计算部分----------------------------
        q = self.q_proj_normal(query) * self.scaling
        k = self.k_proj_normal(query)
        v = self.v_proj_normal(query)
        # 此处变为了[batch_size*num_heads，查询的个数，num_hiddens/num_heads]
        q = self._shape(q, tgt_len, bsz, self.normal_heads)
        if k is not None:
            k = self._shape(k, -1, bsz, self.normal_heads)
        if v is not None:
            v = self._shape(v, -1, bsz, self.normal_heads)


        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.normal_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.normal_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.normal_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.normal_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.normal_heads, tgt_len, src_len)
        attn_weights_normal = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights_normal, p=self.dropout, training=self.training,)

        assert v is not None
        attn_output_normal = torch.bmm(attn_probs, v)
        assert attn_output_normal.size() == (bsz * self.normal_heads, tgt_len, self.head_dim)
        # attn_output_normal = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)


        # =================================Special Attention 计算: MG1=====================
        if self.head_list[0]!=0:
            q = self.q_proj_mg1(query) * self.scaling
            k_mg_1 = self.k_proj_mg1(MG_all_list[0]["key"])
            v_mg_1 = self.v_proj_mg1(MG_all_list[0]["value"])
            q = self._shape(q, tgt_len, bsz, MG_all_list[0]["head_num"])
            if k_mg_1 is not None:
                k = self._shape(k_mg_1, -1, bsz, MG_all_list[0]["head_num"])
            if v_mg_1 is not None:
                v = self._shape(v_mg_1, -1, bsz, MG_all_list[0]["head_num"])
            assert k is not None
            src_len = k.size(1)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert attn_weights.size() == (bsz * MG_all_list[0]["head_num"], tgt_len, src_len)
            key_padding_mask = MG_all_list[0]["padding_mask"].to(device=k.device)

            # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
            if key_padding_mask is not None and key_padding_mask.dim() == 0:
                key_padding_mask = None
            assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

            if key_padding_mask is not None:  # don't attend to padding symbols
                attn_weights = attn_weights.view(bsz, MG_all_list[0]["head_num"], tgt_len, src_len)
                reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
                reshaped = torch.repeat_interleave(reshaped, tgt_len, dim=2)
                attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
                attn_weights = attn_weights.view(bsz * MG_all_list[0]["head_num"], tgt_len, src_len)
            attn_weights_special_1 = F.softmax(attn_weights, dim=-1)
            attn_probs = F.dropout(attn_weights_special_1, p=self.dropout, training=self.training,)

            assert v is not None
            attn_output_special_1 = torch.bmm(attn_probs, v)
            assert attn_output_special_1.size() == (bsz * MG_all_list[0]["head_num"], tgt_len, self.head_dim)

        # =================================Special Attention 计算: MG2=====================
        if self.head_list[1]!=0:
            q = self.q_proj_mg2(query) * self.scaling
            k_mg_2 = self.k_proj_mg2(MG_all_list[1]["key"])
            v_mg_2 = self.v_proj_mg2(MG_all_list[1]["value"])
            q = self._shape(q, tgt_len, bsz, MG_all_list[1]["head_num"])
            if k_mg_2 is not None:
                k = self._shape(k_mg_2, -1, bsz, MG_all_list[1]["head_num"])
            if v_mg_2 is not None:
                v = self._shape(v_mg_2, -1, bsz, MG_all_list[1]["head_num"])
            assert k is not None
            src_len = k.size(1)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert attn_weights.size() == (bsz * MG_all_list[1]["head_num"], tgt_len, src_len)
            key_padding_mask = MG_all_list[1]["padding_mask"].to(device=k.device)

            # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
            if key_padding_mask is not None and key_padding_mask.dim() == 0:
                key_padding_mask = None
            assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

            if key_padding_mask is not None:  # don't attend to padding symbols
                attn_weights = attn_weights.view(bsz, MG_all_list[1]["head_num"], tgt_len, src_len)
                reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
                reshaped = torch.repeat_interleave(reshaped, tgt_len, dim=2)
                attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
                attn_weights = attn_weights.view(bsz * MG_all_list[1]["head_num"], tgt_len, src_len)
            attn_weights_special_2 = F.softmax(attn_weights, dim=-1)
            attn_probs = F.dropout(attn_weights_special_2, p=self.dropout, training=self.training,)

            assert v is not None
            attn_output_special_2 = torch.bmm(attn_probs, v)
            assert attn_output_special_2.size() == (bsz * MG_all_list[1]["head_num"], tgt_len, self.head_dim)

        # =================================Special Attention 计算: MG3=====================
        if self.head_list[2]!=0:
            q = self.q_proj_mg3(query) * self.scaling
            k_mg_3 = self.k_proj_mg3(MG_all_list[2]["key"])
            v_mg_3 = self.v_proj_mg3(MG_all_list[2]["value"])
            q = self._shape(q, tgt_len, bsz, MG_all_list[2]["head_num"])
            if k_mg_3 is not None:
                k = self._shape(k_mg_3, -1, bsz, MG_all_list[2]["head_num"])
            if v_mg_3 is not None:
                v = self._shape(v_mg_3, -1, bsz, MG_all_list[2]["head_num"])
            assert k is not None
            src_len = k.size(1)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert attn_weights.size() == (bsz * MG_all_list[2]["head_num"], tgt_len, src_len)
            key_padding_mask = MG_all_list[2]["padding_mask"].to(device=k.device)

            # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
            if key_padding_mask is not None and key_padding_mask.dim() == 0:
                key_padding_mask = None
            assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

            if key_padding_mask is not None:  # don't attend to padding symbols
                attn_weights = attn_weights.view(bsz, MG_all_list[2]["head_num"], tgt_len, src_len)
                reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
                reshaped = torch.repeat_interleave(reshaped, tgt_len, dim=2)
                attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
                attn_weights = attn_weights.view(bsz * MG_all_list[2]["head_num"], tgt_len, src_len)
            attn_weights_special_3 = F.softmax(attn_weights, dim=-1)
            attn_probs = F.dropout(attn_weights_special_3, p=self.dropout, training=self.training,)

            assert v is not None
            attn_output_special_3 = torch.bmm(attn_probs, v)
            assert attn_output_special_3.size() == (bsz * MG_all_list[2]["head_num"], tgt_len, self.head_dim)

        # =================================Special Attention 计算: MG34=====================
        if self.head_list[3] != 0:
            q = self.q_proj_mg4(query) * self.scaling
            k_mg_4 = self.k_proj_mg4(MG_all_list[3]["key"])
            v_mg_4 = self.v_proj_mg4(MG_all_list[3]["value"])
            q = self._shape(q, tgt_len, bsz, MG_all_list[3]["head_num"])
            if k_mg_4 is not None:
                k = self._shape(k_mg_4, -1, bsz, MG_all_list[3]["head_num"])
            if v_mg_4 is not None:
                v = self._shape(v_mg_4, -1, bsz, MG_all_list[3]["head_num"])
            assert k is not None
            src_len = k.size(1)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert attn_weights.size() == (bsz * MG_all_list[3]["head_num"], tgt_len, src_len)
            key_padding_mask = MG_all_list[3]["padding_mask"].to(device=k.device)

            # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
            if key_padding_mask is not None and key_padding_mask.dim() == 0:
                key_padding_mask = None
            assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

            if key_padding_mask is not None:  # don't attend to padding symbols
                attn_weights = attn_weights.view(bsz, MG_all_list[3]["head_num"], tgt_len, src_len)
                reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
                reshaped = torch.repeat_interleave(reshaped, tgt_len, dim=2)
                attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
                attn_weights = attn_weights.view(bsz * MG_all_list[3]["head_num"], tgt_len, src_len)
            attn_weights_special_4 = F.softmax(attn_weights, dim=-1)
            attn_probs = F.dropout(attn_weights_special_4, p=self.dropout, training=self.training, )

            assert v is not None
            attn_output_special_4 = torch.bmm(attn_probs, v)
            assert attn_output_special_4.size() == (bsz * MG_all_list[3]["head_num"], tgt_len, self.head_dim)

        # ==============================计算总和===================================
        if self.head_list[0]!=0 and self.head_list[1]!=0 and self.head_list[2]!=0 and self.head_list[3]!=0:
            all_attn_output = torch.cat((attn_output_normal, attn_output_special_1, attn_output_special_2,
                                         attn_output_special_3, attn_output_special_4))
        elif self.head_list[0]!=0 and self.head_list[1]!=0 and self.head_list[2]!=0 :
            all_attn_output = torch.cat((attn_output_normal, attn_output_special_1, attn_output_special_2,
                                         attn_output_special_3))
        elif self.head_list[0]!=0 and self.head_list[1]!=0:
            all_attn_output = torch.cat((attn_output_normal, attn_output_special_1, attn_output_special_2))
        elif self.head_list[0]!=0:
            all_attn_output = torch.cat((attn_output_normal, attn_output_special_1))
        else:
            all_attn_output = attn_output_normal
        # all_attn_output = torch.cat((attn_output_normal,attn_output_special_1,attn_output_special_2,attn_output_special_3,attn_output_special_4))
        all_attn_output = all_attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(all_attn_output)
        if need_weights:
            attn_weights = attn_weights_normal.view(bsz, self.normal_heads, tgt_len, tgt_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
        key_padding_mask = self._cat_prev_key_padding_mask(
            key_padding_mask, prev_key_padding_mask, bsz, k.size(1), static_kv
        )
        return k, v, key_padding_mask

    @staticmethod
    def _cat_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)

        elif key_padding_mask is not None:
            filler = torch.zeros(
                batch_size,
                src_len - key_padding_mask.size(1),
                dtype=key_padding_mask.dtype,
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask