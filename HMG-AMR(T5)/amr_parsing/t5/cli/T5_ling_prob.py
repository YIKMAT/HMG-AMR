import warnings
import argparse

import penman
import torch
from amrlib.utils.logging import silence_penman
from penman.models.noop import NoOpModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from amr_utils.datasets.dataset import AMRPenman,  add_prefix, AMR_GENERATION
from amr_parsing.t5.cli.inference import Inference
from amr_parsing.t5.models.lg import LG
from amr_parsing.t5.models.utils import prepare_data
from amr_parsing.t5.models.utils import prepare_data_original
from amr_parsing.t5.cli import ROOT
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.simplefilter('ignore')
import senteval
import os
import yaml





PATH_TO_DATA = 'data'
# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 10}
# 'optim': 'adam'

def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    # 得到了输入的id以及pad batch:(id, pad)
    input_ids, attention_mask = prepare_data_original(sentences, params.tokenize, 512)
    # 获得输入句子token的embedding。(batch_size, seq_num, embedding)
    with torch.no_grad():
        embeddings = params.T5(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 decoder_input_ids=None,
                                 decoder_attention_mask=None,
                                 labels=None,
                                 return_dict=True)
    # 计算出句子的embedding
    senten_lenth = attention_mask.sum(dim=1).unsqueeze(1)
    sen_emb =embeddings[0].sum(dim=1)
    sen_emb = sen_emb / senten_lenth.expand_as(sen_emb)
    sen_emb = sen_emb.data.cpu().numpy()
    return sen_emb

def prepare(params, samples):
    # params.bart.build_vocab([' '.join(s) for s in samples], tokenize=False)
    print("进入了prepare函数")

if __name__ == '__main__':

    # 设置TRANSFORMERS_OFFLINE环境变量为1
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    parser = argparse.ArgumentParser(description='AMR parse')

    parser.add_argument(
        '-m', '--model', type=str, default='t5-base', help='model name')
    parser.add_argument(
        '--max_source_length', type=int, default=16, help='Max source length')
    parser.add_argument(
        '--max_target_length', type=int, default=16, help='Max target length')
    parser.add_argument(
        '--model_type', type=str, default='t5', help='Model type: bart or t5')
    parser.add_argument(
        '-c', '--checkpoint', type=str, default=None, help='Checkpoint model')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/sweeped.yaml',
        help='Use the following config for hparams.')


    args = parser.parse_args()

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)


    net = LG(args.model,
             max_source_length=args.max_source_length,
             max_target_length=args.max_target_length,
             model_type=args.model_type,
             head_list = config['head_list'],
             mg_layer_begin_index = config['mg_layer_begin_index'],
             mg_layer_end_index = config['mg_layer_end_index'],
             mg_list = config['mg_list'],
             hie_layer_begin_index = config['hie_layer_begin_index'],
             hie_layer_end_index = config['hie_layer_end_index'],
             hie_mode = config['hie_mode'],
             lstm_state = config['lstm_state'],
             )

    if args.checkpoint is not None:
        print("Load model from ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        # if torch.cuda.device_count() <= 1:
        net.model.load_state_dict(checkpoint['model_state_dict'])
        # else:
        #     net.model.module.load_state_dict(checkpoint['model_state_dict'])

        # ==========SentEval==========
    params_senteval['T5'] = net.model
    params_senteval['tokenize'] = net.tokenizer
    params_senteval['device'] = net.device
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # transfer_tasks = ['Length']
    transfer_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)


