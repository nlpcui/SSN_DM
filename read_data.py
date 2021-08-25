import torch, sys, json, os, copy
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
from utils import binary_search
import numpy as np
from tqdm import tqdm


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

processed_pubmed_dir = 'dataset/processed/pubmed'
processed_arXiv_dir = 'dataset/processed/arXiv'

CLS_ID = 101
SEP_ID = 102
PAD_ID = 0


class Sentence:
    def __init__(self) -> None:
        self.token_ids = []
        self.token_types = []
        self.attention_mask = []
        self.label = 2
        self.sentence_id = -1
        self.sentence_len = 0

    def to_dict(self):
        return self.__dict__


class Segment:
    def __init__(self, max_length) -> None:
        self.doc_id = ''
        self.segment_id = -1
        self.sentences = []
        self.token_ids = []
        self.token_types = []
        self.position_ids = []
        self.attention_mask = []
        self.token_section_ids = []
        self.token_labels = []
        self.label_indices = []
        self.valid_length = 0
        self.label_num = 0
        self.max_length = max_length
    
    @staticmethod
    def pad_segment(doc_id, segment_id, max_seg_length=512):
        segment = {
            'doc_id': doc_id,
            'segment_id': segment_id,
            'token_ids': [PAD_ID for i in range(max_seg_length)],
            'token_types': [PAD_ID for i in range(max_seg_length)],
            'position_ids': [i for i in range(max_seg_length)],
            'attention_mask': [1 for i in range(max_seg_length)],
            'token_section_ids': [PAD_ID for i in range(max_seg_length)],
            'token_labels': [2 for i in range(max_seg_length)],
            'label_indices': [False for i in range(max_seg_length)],
            'label_num': 0,
        }

        return segment

    def to_dict(self, labels, section_lengths, pad=True):
        '''
        section_length:
        sentence_labels:
        section_length: [sec_length: sec_name, ]
        '''
        # merge sentences
        for sentence in self.sentences:
            self.token_ids.extend(sentence.token_ids)
            self.token_types.extend(sentence.token_types)
            self.attention_mask.extend(sentence.attention_mask)
            
            # section info
            section_id = self.get_section(sentence.sentence_id, section_lengths)
            self.token_section_ids.extend([section_id for i in range(sentence.sentence_len)])

            # labels
            token_labels = [labels[sentence.sentence_id]]
            token_labels.extend([2 for i in range(sentence.sentence_len-1)])
            self.token_labels.extend(token_labels)
            
            self.label_indices.append(True)
            self.label_indices.extend([False for i in range(len(sentence.token_ids)-1)])

        self.position_ids = [i for i in range(self.max_length)]
        self.label_num = len(self.label_indices)

        self.pad()

        seg_dict = {
            'doc_id': self.doc_id,
            'segment_id': self.segment_id,
            'token_ids': self.token_ids,
            'position_ids': self.position_ids,
            'token_types': self.token_types,
            'attention_mask': self.attention_mask,
            'token_section_ids': self.token_section_ids,
            'token_labels': self.token_labels,
            'label_indices': self.label_indices,
            'label_num': self.label_num
        }

        return seg_dict


    def get_section(self, sentence_id, section_lengths):
        cur_len = 0
        for section_id, section_len in section_lengths:
            if sentence_id < cur_len + section_len:
                return section_id
            cur_len += section_len

    def pad(self):
        if self.valid_length < self.max_length:
            self.token_ids.extend([PAD_ID for i in range(self.max_length-self.valid_length)])
            self.attention_mask.extend([PAD_ID for i in range(self.max_length-self.valid_length)])
            self.token_types.extend([PAD_ID for i in range(self.max_length-self.valid_length)])
            self.token_section_ids.extend([PAD_ID for i in range(self.max_length-self.valid_length)])
            self.token_labels.extend([PAD_ID for i in range(self.max_length-self.valid_length)])
            
        self.label_indices.extend([False for i in range(self.max_length-len(self.label_indices))])


class SciSumDataset(Dataset):
    def __init__(self, inputs_dir, labels_dir, references_dir, max_seg_num, max_seg_len, name_, mode) -> None:
        assert mode in ['train', 'val', 'test']
        super(SciSumDataset, self).__init__()
        self.name = name_
        self.doc_ids = []

        self.section_dict = {}
        self.max_seg_num = max_seg_num
        self.max_seg_len = max_seg_len

        self.inputs_dir = inputs_dir
        self.labels_dir = labels_dir
        self.references_dir = references_dir

        input_files = [fn.split('.')[0] for fn in os.listdir(self.inputs_dir)]
        label_files = [fn.split('.')[0] for fn in os.listdir(self.labels_dir)]
        references_files = [fn.split('.')[0] for fn in os.listdir(self.references_dir)]

        self.doc_ids = list(set(input_files) & set(label_files) & set(references_files))
        self.doc_ids.sort()

        self.encoded_doc_id = {self.doc_ids[i]: i for i in range(len(self.doc_ids))}

    def merge_sentences(self, sentence_lst):
        merged_content = []
        for sent in sentence_lst:
            merged_content.extend(sent)
        return merged_content

    def __len__(self):
        return len(self.doc_ids)


    def __getitem__(self, idx):
        doc_id = self.doc_ids[idx]
        
        input_file = os.path.join(self.inputs_dir, doc_id+'.json')
        label_file = os.path.join(self.labels_dir, doc_id+'.json')
        reference_file = os.path.join(self.references_dir, doc_id+'.txt')
        
        doc_word_count = 0

        with open(input_file, 'r') as fp:
            doc = json.loads(fp.readlines()[0])
            section_lengths = [
                (self.section2id(doc['section_names'][i]), doc['section_lengths'][i])
                for i in range(len(doc['section_names']))
            ]

        # for sent in doc['inputs']:
        #     doc_word_count += sent['word_count']
        # print(doc_word_count)

        with open(label_file, 'r') as fp:
            labels = json.loads(fp.readlines()[0])['labels']
        
        with open(reference_file, 'r') as fp:
            ref = fp.readlines()[0]
        
        sentences = []
        cur_sentence = Sentence()

        for sid, sent in enumerate(doc['inputs']):
            encoded = tokenizer.encode_plus(sent['text'], add_special_tokens=True, max_length=512, truncation=True)

            cur_sentence.sentence_id = sid
            cur_sentence.token_ids = encoded['input_ids']
            cur_sentence.token_types = [sid%2 for i in range(len(encoded['input_ids']))]
            cur_sentence.attention_mask = encoded['attention_mask']
            cur_sentence.sentence_len = len(encoded['input_ids'])

            sentences.append(copy.deepcopy(cur_sentence))
            cur_sentence = Sentence()


        segments = self.split_doc(doc['id'], sentences, labels, section_lengths)
        return segments


    def split_doc(self, doc_id, sentences, labels, section_lengths):

        segments = []
        cur_segment_id = 0

        cur_segment = Segment(max_length=self.max_seg_len)
        cur_segment.doc_id = doc_id
        cur_segment.segment_id = cur_segment_id

        for sentence in sentences:
            if cur_segment.valid_length + sentence.sentence_len > self.max_seg_len:
                segments.append(cur_segment.to_dict(labels, section_lengths))
                cur_segment_id += 1
                cur_segment = Segment(max_length=self.max_seg_len)
                cur_segment.doc_id = doc_id
                cur_segment.segment_id = cur_segment_id

            cur_segment.sentences.append(sentence)
            cur_segment.valid_length += sentence.sentence_len

        if cur_segment.valid_length > 0:
            segments.append(cur_segment.to_dict(labels, section_lengths))

        for seg_id in range(len(segments), self.max_seg_num):
            segments.append(Segment.pad_segment(doc_id, seg_id))

        segments = segments[:self.max_seg_num]

        segments = {
            'doc_ids': [segment['doc_id'] for segment in segments],
            'token_ids': np.array([segment['token_ids'] for segment in segments]),
            'segment_ids': np.array([segment['segment_id'] for segment in segments]),
            'token_types': np.array([segment['token_types'] for segment in segments]),
            'position_ids': np.array([segment['position_ids'] for segment in segments]),
            'attention_mask': np.array([segment['attention_mask'] for segment in segments]),
            'token_section_ids': np.array([segment['token_section_ids'] for segment in segments]),
            'token_labels': np.array([segment['token_labels'] for segment in segments]),
            'label_indices': np.array([segment['label_indices'] for segment in segments]),
            'label_num': np.array([segment['label_num'] for segment in segments])
        }

        return segments


    def section2id(self, section_name):
        if section_name not in self.section_dict:
            self.section_dict[section_name] = len(self.section_dict)
        return self.section_dict[section_name]



    def get_instance_by_id(self, num_id):
        if num_id not in self.encoded_doc_id:
            return None, None, None
        
        doc_id = self.encoded_doc_id[num_id]

        with open(os.path.join(self.inputs_dir, doc_id+'.json'), 'r') as fp:
            doc = json.loads(fp.readlines()[0])
        with open(os.path.join(self.labels_dir, doc_id+'.json'), 'r') as fp:
            label = json.loads(fp.readlines()[0])
        with open(os.path.join(self.references_dir, doc_id+'.txt'), 'r') as fp:
            reference = fp.readlines()[0]

        return doc, label, reference


def doc_statistics(base_path):

    inputs_dir = os.path.join(base_path, 'inputs')
    input_files = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(inputs_dir, split)
        for fn in os.listdir(split_dir):
            input_files.append(os.path.join(split_dir, fn))

    section_dict = {}
    
    doc_total_lengths = [] 
    doc_sent_num = []
    doc_sent_lengths = []

    for fn in tqdm(input_files):
        with open(fn, 'r') as fp:
            doc = json.loads(fp.readlines()[0])

        doc_length = 0
        for sentence in doc['inputs']:
            doc_length += sentence['word_count']
            doc_sent_lengths.append(sentence['word_count'])
        
        doc_sent_num.append(len(doc['inputs']))
        doc_total_lengths.append(doc_length)

        for section_name in doc['section_names']:
            if section_name not in section_dict:
                section_dict[section_name] = len(section_dict)
                
    doc_total_lengths.sort()
    doc_sent_num.sort()
    doc_sent_lengths.sort()

    print('average_length: {}, max_length: {}, min_length: {}, median_length: {}'.format(
        sum(doc_total_lengths) / len(doc_total_lengths),
        max(doc_total_lengths),
        min(doc_total_lengths),
        doc_total_lengths[len(doc_total_lengths)//2]
    ))
    print('average_sent_num: {}, max_sent_num: {}, min_sent_num: {}, median_sent_num: {}'.format(
        sum(doc_sent_num) / len(doc_sent_num),
        max(doc_sent_num),
        min(doc_sent_num),
        doc_sent_num[len(doc_sent_num)//2]
    ))
    print('average_sent_length: {}, max_sent_length: {}, min_sent_length: {}, median_sent_length: {}'.format(
        sum(doc_sent_lengths) / len(doc_sent_lengths),
        max(doc_sent_lengths),
        min(doc_sent_lengths),
        doc_sent_lengths[len(doc_sent_lengths)//2]
    ))
    print(section_dict)



if __name__ == '__main__':
    pass
    # doc_statistics(processed_pubmed_dir)
    pubmed_train_dataset = SciSumDataset(processed_pubmed_dir, 'pubmed', 'train')
    print(pubmed_train_dataset[100]['label_indices'])