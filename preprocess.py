import sys
sys.path.append('../')
from paths import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--clip_msra',action='store_true')

args = parser.parse_args()

lexicon_f = open(yangjie_rich_pretrain_word_path,'r')
char_f = open(yangjie_rich_pretrain_unigram_path,'r')

output_f = open(yangjie_rich_pretrain_char_and_word_path,'w')

lexicon_lines = lexicon_f.readlines()
for l in lexicon_lines:
    l_split = l.strip().split()
    if len(l_split[0]) != 1:
        print(l.strip(),file=output_f)

char_lines = char_f.readlines()
for l in char_lines:
    print(l.strip(),file=output_f)

def create_cliped_file(fp):
    f = open(fp,'r',encoding='utf-8')
    fp_out = fp + '_clip2'
    f_out = open(fp_out,'w',encoding='utf-8')
    now_example_len = 0
    # cliped_corpus = [[]]
    # now_example = cliped_corpus[0]

    lines = f.readlines()
    last_line_split = ['','']
    for line in lines:
        line_split = line.strip().split()

        print(line,end='',file=f_out)
        now_example_len += 1
        if len(line_split) == 0 or \
                (line_split[0] in ['。','！','？']
                 and line_split[1] == 'O' and now_example_len>210):
            print('',file=f_out)
            now_example_len = 0
        elif ((line_split[0] in ['，','；'] or (now_example_len>1 and last_line_split[0] == '…' and line_split[0] == '…'))
                 and line_split[1] == 'O' and now_example_len>210):
            print('',file=f_out)
            now_example_len = 0

        elif line_split[1][0].lower() == 'e' and now_example_len>210:
            print('',file=f_out)
            now_example_len = 0

        last_line_split = line_split

    f_out.close()
    f_check = open(fp_out,'r',encoding='utf-8')
    lines = f_check.readlines()
    cliped_examples = [[]]
    now_example = cliped_examples[0]
    for line in lines:
        line_split = line.strip().split()
        if len(line_split) == 0:
            cliped_examples.append([])
            now_example = cliped_examples[-1]
        else:
            now_example.append(line.strip())

    check = 0
    max_length = 0
    for example in cliped_examples:
        if len(example)>200:
            print(len(example),''.join(map(lambda x:x.split(' ')[0],example)))
            check = 1

        max_length = max(max_length,len(example))

    print('最长的句子有:{}'.format(max_length))

    if check == 0:
        print('没句子超过200的长度')


if args.clip_msra:
    create_cliped_file('{}/train_dev.char.bmes'.format(msra_ner_cn_path))
    create_cliped_file('{}/test.char.bmes'.format(msra_ner_cn_path))



