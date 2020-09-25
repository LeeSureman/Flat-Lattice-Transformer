from fastNLP import cache_results


@cache_results(_cache_fp='need_to_defined_fp',_refresh=True)
def equip_chinese_ner_with_lexicon(datasets,vocabs,embeddings,w_list,word_embedding_path=None,
                                   only_lexicon_in_train=False,word_char_mix_embedding_path=None,
                                   number_normalized=False,
                                   lattice_min_freq=1,only_train_min_freq=0):
    from fastNLP.core import Vocabulary
    def normalize_char(inp):
        result = []
        for c in inp:
            if c.isdigit():
                result.append('0')
            else:
                result.append(c)

        return result

    def normalize_bigram(inp):
        result = []
        for bi in inp:
            tmp = bi
            if tmp[0].isdigit():
                tmp = '0'+tmp[:1]
            if tmp[1].isdigit():
                tmp = tmp[0]+'0'

            result.append(tmp)
        return result

    if number_normalized == 3:
        for k,v in datasets.items():
            v.apply_field(normalize_char,'chars','chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                no_create_entry_dataset=[datasets['dev'], datasets['test']])

        for k,v in datasets.items():
            v.apply_field(normalize_bigram,'bigrams','bigrams')
        vocabs['bigram'] = Vocabulary()
        vocabs['bigram'].from_dataset(datasets['train'], field_name='bigrams',
                                  no_create_entry_dataset=[datasets['dev'], datasets['test']])


    if only_lexicon_in_train:
        print('已支持只加载在trian中出现过的词汇')

    def get_skip_path(chars, w_trie):
        sentence = ''.join(chars)
        result = w_trie.get_lexicon(sentence)
        # print(result)

        return result
    from V0.utils_ import Trie
    from functools import partial
    from fastNLP.core import Vocabulary
    # from fastNLP.embeddings import StaticEmbedding
    from fastNLP_module import StaticEmbedding
    from fastNLP import DataSet
    a = DataSet()
    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)


    if only_lexicon_in_train:
        lexicon_in_train = set()
        for s in datasets['train']['chars']:
            lexicon_in_s = w_trie.get_lexicon(s)
            for s,e,lexicon in lexicon_in_s:
                lexicon_in_train.add(''.join(lexicon))

        print('lexicon in train:{}'.format(len(lexicon_in_train)))
        print('i.e.: {}'.format(list(lexicon_in_train)[:10]))
        w_trie = Trie()
        for w in lexicon_in_train:
            w_trie.insert(w)






    import copy
    for k,v in datasets.items():
        v.apply_field(partial(get_skip_path,w_trie=w_trie),'chars','lexicons')
        v.apply_field(copy.copy, 'chars','raw_chars')
        v.add_seq_len('lexicons','lex_num')
        v.apply_field(lambda x:list(map(lambda y: y[0], x)), 'lexicons', 'lex_s')
        v.apply_field(lambda x: list(map(lambda y: y[1], x)), 'lexicons', 'lex_e')


    if number_normalized == 1:
        for k,v in datasets.items():
            v.apply_field(normalize_char,'chars','chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                no_create_entry_dataset=[datasets['dev'], datasets['test']])

    if number_normalized == 2:
        for k,v in datasets.items():
            v.apply_field(normalize_char,'chars','chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                no_create_entry_dataset=[datasets['dev'], datasets['test']])

        for k,v in datasets.items():
            v.apply_field(normalize_bigram,'bigrams','bigrams')
        vocabs['bigram'] = Vocabulary()
        vocabs['bigram'].from_dataset(datasets['train'], field_name='bigrams',
                                  no_create_entry_dataset=[datasets['dev'], datasets['test']])


    def concat(ins):
        chars = ins['chars']
        lexicons = ins['lexicons']
        result = chars + list(map(lambda x:x[2],lexicons))
        # print('lexicons:{}'.format(lexicons))
        # print('lex_only:{}'.format(list(filter(lambda x:x[2],lexicons))))
        # print('result:{}'.format(result))
        return result

    def get_pos_s(ins):
        lex_s = ins['lex_s']
        seq_len = ins['seq_len']
        pos_s = list(range(seq_len)) + lex_s

        return pos_s

    def get_pos_e(ins):
        lex_e = ins['lex_e']
        seq_len = ins['seq_len']
        pos_e = list(range(seq_len)) + lex_e

        return pos_e



    for k,v in datasets.items():
        v.apply(concat,new_field_name='lattice')
        v.set_input('lattice')
        v.apply(get_pos_s,new_field_name='pos_s')
        v.apply(get_pos_e, new_field_name='pos_e')
        v.set_input('pos_s','pos_e')

    # print(list(datasets['train'][:10]['lexicons']))
    # print(list(datasets['train'][:10]['lattice']))
    # print(list(datasets['train'][:10]['lex_s']))
    # print(list(datasets['train'][:10]['lex_e']))
    # print(list(datasets['train'][:10]['pos_s']))
    # print(list(datasets['train'][:10]['pos_e']))
    # exit(1208)


    word_vocab = Vocabulary()
    word_vocab.add_word_lst(w_list)
    vocabs['word'] = word_vocab

    lattice_vocab = Vocabulary()
    lattice_vocab.from_dataset(datasets['train'],field_name='lattice',
                               no_create_entry_dataset=[v for k,v in datasets.items() if k != 'train'])
    vocabs['lattice'] = lattice_vocab
    # for k,v in datasets.items():
    #     v.apply_field(lambda x:[ list(map(lambda x:x[0],p)) for p in x],'skips_l2r','skips_l2r_source')
    #     v.apply_field(lambda x:[ list(map(lambda x:x[1],p)) for p in x], 'skips_l2r', 'skips_l2r_word')
    #
    # for k,v in datasets.items():
    #     v.apply_field(lambda x:[ list(map(lambda x:x[0],p)) for p in x],'skips_r2l','skips_r2l_source')
    #     v.apply_field(lambda x:[ list(map(lambda x:x[1],p)) for p in x], 'skips_r2l', 'skips_r2l_word')

    # for k,v in datasets.items():
    #     v.apply_field(lambda x:list(map(len,x)), 'skips_l2r_word', 'lexicon_count')
    #     v.apply_field(lambda x:
    #                   list(map(lambda y:
    #                            list(map(lambda z:word_vocab.to_index(z),y)),x)),
    #                   'skips_l2r_word',new_field_name='skips_l2r_word')
    #
    #     v.apply_field(lambda x:list(map(len,x)), 'skips_r2l_word', 'lexicon_count_back')
    #
    #     v.apply_field(lambda x:
    #                   list(map(lambda y:
    #                            list(map(lambda z:word_vocab.to_index(z),y)),x)),
    #                   'skips_r2l_word',new_field_name='skips_r2l_word')





    if word_embedding_path is not None:
        word_embedding = StaticEmbedding(word_vocab,word_embedding_path,word_dropout=0)
        embeddings['word'] = word_embedding

    if word_char_mix_embedding_path is not None:
        lattice_embedding = StaticEmbedding(lattice_vocab, word_char_mix_embedding_path,word_dropout=0.01,
                                            min_freq=lattice_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['lattice'] = lattice_embedding

    vocabs['char'].index_dataset(* (datasets.values()),
                             field_name='chars', new_field_name='chars')
    vocabs['bigram'].index_dataset(* (datasets.values()),
                               field_name='bigrams', new_field_name='bigrams')
    vocabs['label'].index_dataset(* (datasets.values()),
                              field_name='target', new_field_name='target')
    vocabs['lattice'].index_dataset(* (datasets.values()),
                                    field_name='lattice', new_field_name='lattice')


    return datasets,vocabs,embeddings