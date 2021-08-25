import rouge, tqdm
import os, sys, json


rouger = rouge.Rouge()

def compute_oracle():
    oracle_rouge_1 = 0
    oracle_rouge_2 = 0
    oracle_rouge_l = 0

    file_list = os.listdir('./dataset/inputs')
    file_list = [file_name.split('.')[0] for file_name in file_list]
    
    for file_name in file_list:
        with open(os.path.join('dataset/inputs', file_name+'.json')) as fp:
            doc = json.loads(fp.readlines()[0])
        with open(os.path.join('dataset/labels', file_name+'.json')) as fp:
            labels = json.loads(fp.readlines()[0])['labels']
        with open(os.path.join('dataset/references', file_name+'.txt')) as fp:
            ref = fp.readlines()[0]
        
        golden_sentences = []
        for i in range(len(doc['inputs'])):
            if labels[i] == 1:
                golden_sentences.append(doc['inputs'][i]['text'])

        rouge_score = rouger.get_scores(' '.join(golden_sentences), ref)
        # print('doc id {}, rouge score {}'.format(doc['id'], rouge_score))

        oracle_rouge_1 += rouge_score[0]['rouge-1']['f']
        oracle_rouge_2 += rouge_score[0]['rouge-2']['f']
        oracle_rouge_l += rouge_score[0]['rouge-l']['f']

    print('Oracle: rouge_1: {}, rouge_2: {}, rouge_l: {}'.format(oracle_rouge_1, oracle_rouge_2, oracle_rouge_l))





def eval_model(src_dir, label_dir, output_dir, ref_dir):

    def get_index(logits, k=6, theshord=-1):
        sorted_indices = sorted(range(len(s)), key=lambda k: s[k])
        n_1 = k
        n_2 = len(logits)
        for i in range(len(sorted_indices)):
            if threshold > 0 and logits[i] < threshold:
                n_2 = i
                break
        n = min(n_1, n_2)

        return sorted_indices[:n]
    
    def get_accuracy(oracle_labels, pred_labels):
        correct_cnt = 0
        wrong_cnt = 0
        for i in range(len(oracle_labels)):
            for j in range(len(oracle_labels[i])):
                if oracle_labels[i][j] == pred_labels[i][j]:
                    correct_cnt += 1
                else:
                    wrong_cnt += 1
        return wrong_cnt / (wrong_cnt + correct_cnt)


    src_files = [fn.split('.')[0] for fn in os.listdir(src_dir)]
    label_files = [fn.split('.')[0] for fn in os.listdir(label_dir)]
    output_files = [fn.split('.')[0] for fn in os.listdir(output_dir)]
    ref_files = [fn.split('.')[0] for fn in os.listdir(ref_dir)]

    eval_files = set(label_files) & set(output_files) & set(ref_files)
    print('{} files for evaluation'.format(len(eval_files)))

    rouge_scores = {
        'rouge-1': {'p': [], 'r': [], 'f': []},
        'rouge-2': {'p': [], 'r': [], 'f': []},
        'rouge-l': {'p': [], 'r': [], 'f': []},
    }

    pred_labels = []
    oracle_labels = []

    for fn in tqdm.tqdm(eval_files):
        with open(os.path.join(src_dir, fn+'.json'), 'r') as fp:
            doc = json.loads(fp.readlines()[0])

        with open(os.path.join(label_dir, fn+'.json'), 'r') as fp:
            oracle = json.loads(fp.readlines()[0])
            oracle_label = oracle['labels']
        
        with open(os.path.join(output_dir, fn+'.json'), 'r') as fp:
            out = json.loads(fp.readlines()[0])
            logits = out['logits']
            rank = out['rank']

        with open(os.path.join(ref_dir, fn+'.txt'), 'r') as fp:
            ref_summary = fp.readlines()[0]
        
        pred_label = [0 for i in range(len(oracle_label))] 

        # extract_indices = get_index(logits=logits)
        extract_summary = []
        for index in rank[:6]:
            pred_label[index] = 1
            extract_summary.append(doc['inputs'][index]['text'])
        extract_summary = ' '.join(extract_summary)

        pred_labels.append(pred_label)
        oracle_labels.append(oracle_label)

        try:
            scores = rouger.get_scores(extract_summary, ref_summary)
        except Exception:
            continue
        rouge_scores['rouge-1']['p'].append(scores[0]['rouge-1']['p'])
        rouge_scores['rouge-1']['r'].append(scores[0]['rouge-1']['r'])
        rouge_scores['rouge-1']['f'].append(scores[0]['rouge-1']['f'])

        rouge_scores['rouge-2']['p'].append(scores[0]['rouge-2']['p'])
        rouge_scores['rouge-2']['r'].append(scores[0]['rouge-2']['r'])
        rouge_scores['rouge-2']['f'].append(scores[0]['rouge-2']['f'])

        rouge_scores['rouge-l']['p'].append(scores[0]['rouge-l']['p'])
        rouge_scores['rouge-l']['r'].append(scores[0]['rouge-l']['r'])
        rouge_scores['rouge-l']['f'].append(scores[0]['rouge-l']['f'])

    print('Average: R-1 is {}, R-2 is {}, R-L is {}'.format(
        sum(rouge_scores['rouge-1']['f']) / len(rouge_scores['rouge-1']['f']),
        sum(rouge_scores['rouge-2']['f']) / len(rouge_scores['rouge-2']['f']),
        sum(rouge_scores['rouge-l']['f']) / len(rouge_scores['rouge-l']['f'])
    ))

    accuracy = get_accuracy(oracle_labels, pred_labels)
    
    print('accuracy is {}'.format(accuracy))


output_dir = 'output/pubmed_test' # 'output/arXiv_test' 
src_dir = 'dataset/pubmed/inputs/test' # 'output/arXiv_test' 
label_dir = 'dataset/pubmed/labels/test' # 'output/arXiv_test' 
ref_dir = 'dataset/pubmed/references/test' # 'output/arXiv_test


if __name__ == '__main__':
    eval_model(src_dir, label_dir, output_dir, ref_dir)
    # compute_oracle()
    