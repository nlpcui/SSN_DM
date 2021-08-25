# coding: utf-8
# this file will be executed by main.sh
import enum, time, os
import sys, json, torch, configparser
from tokenizers import models
from read_data import SciSumDataset
from ssn_dm import SSNDM
from torch.utils.data import Dataset, DataLoader
import rouge
import numpy as np
from transformers import (BertModel, BertTokenizer, AutoConfig)
from utils import check_nan
from mlx_studio.storage import hdfs
from tqdm import tqdm

# hdfs.upload("result.txt", output_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import argparse


def early_stop(losses, step=3):
    if len(losses) < step+1:
        return False
    
    l1 = np.array(losses[len(losses)-step:])
    l2 = np.array(losses[len(losses)-step-1:-1])

    return (l1-l2<0).all()


def output_layers(model):
    for name, parameters in model.named_parameters():
        print(name,':',parameters.size())

def output_size(model):
    size = sum(param.numel() for param in netD.parameters())
    print('total size is {}'.format(size)) 




def train_model(args):

    assert args.dataset in ['pubmed', 'arXiv']
    torch.cuda.empty_cache()

    if args.dataset == 'pubmed':
        data_train = SciSumDataset(
            inputs_dir = args.pubmed_train_inputs_dir, 
            labels_dir = args.pubmed_train_labels_dir,
            references_dir = args.pubmed_train_references_dir,
            name_ = 'pubmed',
            mode = 'train',
            max_seg_num=args.max_seg_num, 
            max_seg_len=args.max_seg_len
        )

        data_val = SciSumDataset(
            inputs_dir = args.pubmed_val_inputs_dir, 
            labels_dir = args.pubmed_val_labels_dir,
            references_dir = args.pubmed_val_references_dir,
            name_ = 'pubmed', 
            mode = 'val',
            max_seg_num=args.max_seg_num, 
            max_seg_len=args.max_seg_len
        )
    else:
        data_train = SciSumDataset(
            inputs_dir = args.arXiv_train_inputs_dir, 
            labels_dir = args.arXiv_train_labels_dir,
            references_dir = args.arXiv_train_references_dir,
            name_ = 'arXiv',
            mode = 'train',
            max_seg_num=args.max_seg_num, 
            max_seg_len=args.max_seg_len
        )

        data_val = SciSumDataset(
            inputs_dir = args.arXiv_val_inputs_dir, 
            labels_dir = args.arXiv_val_labels_dir,
            references_dir = args.arXiv_val_references_dir,
            name_ = 'arXiv', 
            mode = 'val',
            max_seg_num=args.max_seg_num,
            max_seg_len=args.max_seg_len
        )


    parallel_batch_size = args.batch_size
    
    if args.parallel and cards_cnt > 0:
        parallel_batch_size *= cards_cnt # scale batch_size for multi-gpu parallel training

    print('cards count {}, parallel batch size is {}'.format(cards_cnt, parallel_batch_size))

    train_data_loader = DataLoader(data_train, batch_size=parallel_batch_size)
    val_data_loader = DataLoader(data_val, batch_size=parallel_batch_size)

    print('build data loader success, {} train data, {} val data'.format(len(data_train), len(data_val)))

    # segment_encoder = BertModel.from_pretrained(config['Setting']['BERT_VERSION'])

    model = SSNDM(args)
    model.train()
    device = 'cpu'
    if args.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if args.parallel and torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
    print('device {}'.format(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    val_losses = []
    stop_epcoh = 0

    for epoch_idx in range(args.epoch):
        epoch_train_loss = 0
        epoch_cls_cnt = 0
        t_start = time.time()
        batch_cnt = len(data_train) // parallel_batch_size
        print('total batch_cnt {}'.format(batch_cnt))
        for batch_idx, batch_data in enumerate(train_data_loader):
            if batch_idx == 20:
                exit(1)
            batch_cls_cnt = 0
            batch_train_loss = 0
            # print(batch_data['doc_ids'])
            segment_position_ids = torch.chunk(batch_data['segment_ids'], args.max_seg_num, 1)
            segment_token_ids = torch.chunk(batch_data['token_ids'], args.max_seg_num, 1)
            segment_token_types = torch.chunk(batch_data['token_types'], args.max_seg_num, 1)
            segment_token_position_ids = torch.chunk(batch_data['position_ids'], args.max_seg_num, 1)
            segment_attention_mask = torch.chunk(batch_data['attention_mask'], args.max_seg_num, 1)
            segment_token_section_ids = torch.chunk(batch_data['token_section_ids'], args.max_seg_num, 1)
            segment_token_labels = torch.chunk(batch_data['token_labels'], args.max_seg_num, 1)
            segment_label_indices = torch.chunk(batch_data['label_indices'], args.max_seg_num, 1)

            memories = [None, ]
            optimizer.zero_grad()
            for segment_id in range(25):
                segment_cls_cnt = torch.sum(segment_label_indices[segment_id]).numpy()
                batch_cls_cnt += segment_cls_cnt
                epoch_cls_cnt += segment_cls_cnt
                if segment_cls_cnt == 0:
                    break

                sent_scores, memory = model(
                    batch_size=args.batch_size,
                    segment_idx=segment_position_ids[segment_id].cuda(), # to(device),
                    token_ids=segment_token_ids[segment_id].cuda(), # to(device),
                    token_types=segment_token_types[segment_id].cuda(), # to(device),
                    position_ids=segment_token_position_ids[segment_id].cuda(), # .to(device),
                    attention_mask=segment_attention_mask[segment_id].cuda(), # to(device),
                    token_section_ids=segment_token_section_ids[segment_id].cuda(), # to(device),
                    label_indices=segment_label_indices[segment_id].cuda(), # to(device),
                    memory=memories[-1] if memories[-1] is None else memories[-1]
                )
                memories.append(memory.cuda()) # to(device))
                sentence_lables = torch.masked_select(segment_token_labels[segment_id].cuda(), segment_label_indices[segment_id].cuda())
                
                loss = criterion(sent_scores, sentence_lables.float())
                # print('sent_scores: {}'.format(sent_scores))
                # print('lables: {}'.format(sentence_lables.float()))
                # print('loss: {}'.format(loss))
                loss.backward()
                
                epoch_train_loss += loss.float().item()
                batch_train_loss += loss.float().item()

            optimizer.step()
            batch_average_loss = batch_train_loss / batch_cls_cnt # token-level

            print('batch_train_loss:{}, batch_cls_cnt:{}'.format(batch_train_loss, batch_cls_cnt))

            t_end = time.time()
            avg_cost = (t_end - t_start)/(batch_idx+1)
            if batch_idx % 1 == 0:
                print('Train: In {} epoch, {} batch, avg loss is {}, {} sec per batch'.format(epoch_idx, batch_idx, batch_average_loss, avg_cost))

        # print('train time cost: {}'.format(t_end - t_start))
        epoch_average_loss = epoch_train_loss / epoch_cls_cnt # token-level 
        print('Train: In {} epoch, avg loss is {}'.format(epoch_idx, epoch_average_loss))

        # eval on validation set
        # val_losses.append(val_model(model, val_data_loader, config))
        # if args.early_stop and early_stop(val_loss):
        #     print('Early stop in {} epoch'.format(val_losses[-1]))
        #     break

        # exit(1)

    model_save_path = '{}_{}'.format(args.save_path, args.dataset)
    torch.save(
        # {
        #     'epoch': stop_epoch+1,
        #     'state_dict': model.state_dict(),
        #     'train_loss': train_loss,
        #     'val_loss': val_loss[:-1]
        # },
        model.state_dict(),
        model_save_path
    )
    hdfs.upload(model_save_path, args.upload_dir)
        


def val_model(model, dataloader, config):

    val_loss = 0
    cnt = 0

    for batch_idx, batch_data in enumerate(train_data):
        t_start = time.time()
        segment_position_ids = torch.chunk(batch_data['segment_ids'], config['Setting'].getint('MAX_SEG_NUM'), 1)
        segment_token_ids = torch.chunk(batch_data['token_ids'], config['Setting'].getint('MAX_SEG_NUM'), 1)
        segment_token_types = torch.chunk(batch_data['token_types'], config['Setting'].getint('MAX_SEG_NUM'), 1)
        segment_token_position_ids = torch.chunk(batch_data['position_ids'], config['Setting'].getint('MAX_SEG_NUM'), 1)
        segment_attention_mask = torch.chunk(batch_data['attention_mask'], config['Setting'].getint('MAX_SEG_NUM'), 1)
        segment_token_section_ids = torch.chunk(batch_data['token_section_ids'], config['Setting'].getint('MAX_SEG_NUM'), 1)
        segment_token_labels = torch.chunk(batch_data['token_labels'], config['Setting'].getint('MAX_SEG_NUM'), 1)
        segment_label_indices = torch.chunk(batch_data['label_indices'], config['Setting'].getint('MAX_SEG_NUM'), 1)

        memories = [None, ]
        batch_cls_cnt = 0

        for segment_id in range(25):
            segment_cls_cnt = torch.sum(segment_label_indices[segment_id]).numpy()
            batch_cls_cnt += segment_cls_cnt
            if segment_cls_cnt == 0:
                break
            logits_cls, memory = model(
                batch_size=config['Setting'].getint('BATCH_SIZE'),
                segment_idx=segment_position_ids[segment_id].contiguous().to(device),
                token_ids=segment_token_ids[segment_id].contiguous().to(device),
                token_types=segment_token_types[segment_id].contiguous().to(device),
                position_ids=segment_token_position_ids[segment_id].contiguous().to(device),
                attention_mask=segment_attention_mask[segment_id].contiguous().to(device),
                token_section_ids=segment_token_section_ids[segment_id].contiguous().to(device),
                label_indices=segment_label_indices[segment_id].contiguous().to(device),
                memory=memories[-1] if memories[-1] is None else memories[-1]
            )
            memories.append(memory.to(device))
            sentence_lables = torch.masked_select(segment_token_labels[segment_id].to(device), segment_label_indices[segment_id].to(device))

            print('segment {} has {} cls'.format(segment_id, sentence_lables.shape[0]))
            loss = criterion(logits_cls, sentence_lables.float())
            val_loss += loss.float().item()

    avg_val_loss = val_loss / cnt 

    return avg_val_loss


def test_model(args):
    assert args.dataset in ['pubmed', 'arXiv']
    if args.dataset == 'pubmed':
        data_test = SciSumDataset(
            inputs_dir = config['Path']['pubmed_test_inputs_dir'], 
            labels_dir = config['Path']['pubmed_test_labels_dir'],
            references_dir = config['Path']['pubmed_test_references_dir'],
            name_ = 'pubmed',
            mode = 'test',
            max_seg_num=config['Setting'].getint('MAX_SEG_NUM'), 
            max_seg_len=config['Setting'].getint('MAX_SEG_LEN')
        )
    else:
        data_test = SciSumDataset(
            inputs_dir = config['Path']['arXiv_test_inputs_dir'], 
            labels_dir = config['Path']['arXiv_test_labels_dir'],
            references_dir = config['Path']['arXiv_test_references_dir'],
            name_ = 'arXiv',
            mode = 'test',
            max_seg_num=config['Setting'].getint('MAX_SEG_NUM'), 
            max_seg_len=config['Setting'].getint('MAX_SEG_LEN')
        )

    print('load test data over, {} instacnes in total'.format(len(data_test)))
    test_dataloader = DataLoader(data_test, batch_size=1)
    model = SSNDM(args)
    model_path = '{}_{}'.format(args.save_path, args.dataset)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()


    for batch_idx, batch_data in tqdm(enumerate(test_dataloader)):
        segment_position_ids = torch.chunk(batch_data['segment_ids'], config['Setting'].getint('MAX_SEG_NUM'), 1)
        segment_token_ids = torch.chunk(batch_data['token_ids'], config['Setting'].getint('MAX_SEG_NUM'), 1)
        segment_token_types = torch.chunk(batch_data['token_types'], config['Setting'].getint('MAX_SEG_NUM'), 1)
        segment_token_position_ids = torch.chunk(batch_data['position_ids'], config['Setting'].getint('MAX_SEG_NUM'), 1)
        segment_attention_mask = torch.chunk(batch_data['attention_mask'], config['Setting'].getint('MAX_SEG_NUM'), 1)
        segment_token_section_ids = torch.chunk(batch_data['token_section_ids'], config['Setting'].getint('MAX_SEG_NUM'), 1)
        segment_token_labels = torch.chunk(batch_data['token_labels'], config['Setting'].getint('MAX_SEG_NUM'), 1)
        segment_label_indices = torch.chunk(batch_data['label_indices'], config['Setting'].getint('MAX_SEG_NUM'), 1)

        memories = [None, ]
        batch_result = []
        for segment_id in range(25):
            if torch.sum(segment_label_indices[segment_id]).numpy() == 0:
                break
            logits_cls, memory = model(
                batch_size=1,
                segment_idx=segment_position_ids[segment_id].cuda(),
                token_ids=segment_token_ids[segment_id].cuda(),
                token_types=segment_token_types[segment_id].cuda(),
                position_ids=segment_token_position_ids[segment_id].cuda(),
                attention_mask=segment_attention_mask[segment_id].cuda(),
                token_section_ids=segment_token_section_ids[segment_id].cuda(),
                label_indices=segment_label_indices[segment_id].cuda(),
                memory=memories[-1] if memories[-1] is None else memories[-1].cuda()
            )
            memories.append(memory)
            logits_cls_sigmoid = torch.nn.functional.sigmoid(logits_cls)
            segment_result = logits_cls_sigmoid.cpu().tolist()
            batch_result.extend(segment_result)
        # print(batch_data['doc_ids'][0][0])
        # print(batch_result)
        rank = sorted(range(len(batch_result)), key=lambda i:batch_result[i], reverse=True)
        # exit(1)
        # print(max(batch_result))
        json_result = {
            'id': batch_data['doc_ids'][0][0],
            'logits': batch_result,
            'rank': rank
        }
        if batch_idx % 100 == 0:
            print('predicted {} instances'.format(batch_idx))
        with open(os.path.join(args.output_dir, batch_data['doc_ids'][0][0]+'.json'), 'w') as fp:
            fp.write(json.dumps(json_result))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--upload_dir', type=str, default='hdfs://haruna/user/cuipeng/data/pcs_guess')
    parser.add_argument('--output_dir', type=str, default='output/pubmed_test')
    parser.add_argument('--memory_hops', type=int, default=3)
    parser.add_argument('--memory_slots', type=int, default=50)
    parser.add_argument('--memory_dim', type=int, default=768)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--max_seg_len', type=int, default=512)
    parser.add_argument('--sect_num', type=int, default=20)
    parser.add_argument('--bert_version', type=str, default='bert-base-uncased')
    parser.add_argument('--max_seg_num', type=int, default=25)
    parser.add_argument('--max_sent_num', type=int, default=500)
    parser.add_argument('--ext_ff_size', type=int, default=2048)
    parser.add_argument('--ext_dropout', type=float, default=0.1)
    parser.add_argument('--ext_head_num', type=int, default=8)
    parser.add_argument('--ext_layer_num', type=int, default=3)
    parser.add_argument('--gat_head_num', type=int, default=6)
    parser.add_argument('--gat_dropout', type=float, default=0.1)


    # training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--early_stop', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='pubmed')
    parser.add_argument('--save_path', type=str, default='model')
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument("--lr", default=1, type=float)
    parser.add_argument("--beta1", default= 0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--warmup_steps", default=8000, type=int)
    parser.add_argument("--warmup_steps_bert", default=8000, type=int)
    parser.add_argument("--warmup_steps_dec", default=8000, type=int)
    parser.add_argument("--max_grad_norm", default=0, type=float)

    # corpus
    ## PubMed train
    parser.add_argument("--pubmed_train_inputs_dir", type=str, default='dataset/pubmed/inputs/test')
    parser.add_argument("--pubmed_train_labels_dir", type=str, default='dataset/pubmed/labels/test')
    parser.add_argument("--pubmed_train_references_dir", type=str, default="dataset/pubmed/references/test")

    ## PubMed val
    parser.add_argument("--pubmed_val_inputs_dir", type=str, default="dataset/pubmed/inputs/test")
    parser.add_argument("--pubmed_val_labels_dir", type=str, default="dataset/pubmed/labels/test")
    parser.add_argument("--pubmed_val_references_dir", type=str, default="dataset/pubmed/references/test")

    ## PubMed test
    parser.add_argument("--pubmed_test_inputs_dir", type=str, default="dataset/pubmed/inputs/test")
    parser.add_argument("--pubmed_test_labels_dir", type=str, default="dataset/pubmed/labels/test")
    parser.add_argument("--pubmed_test_references_dir", type=str, default="dataset/pubmed/references/test")

    ## ArXiv train
    parser.add_argument("--arXiv_train_inputs_dir", type=str, default="dataset/arXiv/inputs/test")
    parser.add_argument("--arXiv_train_labels_dir", type=str, default="dataset/arXiv/labels/test")
    parser.add_argument("--arXiv_train_references_dir", type=str, default="dataset/arXiv/references/test")

    ## ArXiv val
    parser.add_argument("--arXiv_val_inputs_dir", type=str, default="dataset/arXiv/inputs/test")
    parser.add_argument("--arXiv_val_labels_dir", type=str, default="dataset/arXiv/labels/test")
    parser.add_argument("--arXiv_val_references_dir", type=str, default="dataset/arXiv/references/test")

    ## ArXiv test
    parser.add_argument("--arXiv_test_inputs_dir", type=str, default="dataset/arXiv/inputs/test")
    parser.add_argument("--arXiv_test_labels_dir", type=str, default="dataset/arXiv/labels/test")
    parser.add_argument("--arXiv_test_references_dir", type=str, default="dataset/arXiv/references/test")

    args = parser.parse_args()
    cards_cnt = torch.cuda.device_count()
    rouger = rouge.Rouge()


    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)

