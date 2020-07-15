import torch.nn as nn
import torch
from V0.modules import Transformer_Encoder
from fastNLP.modules import ConditionalRandomField
import collections
from utils import get_crf_zero_init
from fastNLP import seq_len_to_mask
from utils import print_info
from utils import better_init_rnn
from fastNLP.modules import LSTM
import math
import copy
from utils import size2MB
from utils import MyDropout

def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    rel pos init:
    如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
    如果是1，那么就按-max_len,max_len来初始化
    """
    num_embeddings = 2*max_seq_len+1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len,max_seq_len+1, dtype=torch.float).unsqueeze(1)*emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb


class Absolute_SE_Position_Embedding(nn.Module):
    def __init__(self,fusion_func,hidden_size,learnable,mode=collections.defaultdict(bool),pos_norm=False,max_len=5000,):
        '''

        :param fusion_func:暂时只有add和concat(直接拼接然后接线性变换)，
        后续得考虑直接拼接再接非线性变换，和将S和E两个位置做非线性变换再加或拼接
        :param hidden_size:
        :param learnable:
        :param debug:
        :param pos_norm:
        :param max_len:
        '''
        super().__init__()
        self.fusion_func = fusion_func
        assert self.fusion_func in {'add','concat','nonlinear_concat','nonlinear_add','add_nonlinear','concat_nonlinear'}
        self.pos_norm = pos_norm
        self.mode = mode
        self.hidden_size = hidden_size
        pe = get_embedding(max_len,hidden_size)
        pe_sum = pe.sum(dim=-1,keepdim=True)
        if self.pos_norm:
            with torch.no_grad():
                pe = pe / pe_sum
        # pe = pe.unsqueeze(0)
        pe_s = copy.deepcopy(pe)
        pe_e = copy.deepcopy(pe)
        self.pe_s = nn.Parameter(pe_s, requires_grad=learnable)
        self.pe_e = nn.Parameter(pe_e, requires_grad=learnable)
        if self.fusion_func == 'concat':
            self.proj = nn.Linear(self.hidden_size * 3,self.hidden_size)

        if self.fusion_func == 'nonlinear_concat':
            self.pos_proj = nn.Sequential(nn.Linear(self.hidden_size * 2,self.hidden_size),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.hidden_size,self.hidden_size))
            self.proj = nn.Linear(self.hidden_size * 2,self.hidden_size)

        if self.fusion_func == 'nonlinear_add':
            self.pos_proj = nn.Sequential(nn.Linear(self.hidden_size * 2,self.hidden_size),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.hidden_size,self.hidden_size))

        if self.fusion_func == 'concat_nonlinear':
            self.proj = nn.Sequential(nn.Linear(self.hidden_size * 3,self.hidden_size),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.hidden_size,self.hidden_size))

        if self.fusion_func == 'add_nonlinear':
            self.proj = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.hidden_size,self.hidden_size))


    def forward(self,inp,pos_s,pos_e):
        batch = inp.size(0)
        max_len = inp.size(1)
        pe_s = self.pe_s.index_select(0, pos_s.view(-1)).view(batch,max_len,-1)
        pe_e = self.pe_e.index_select(0, pos_e.view(-1)).view(batch,max_len,-1)

        if self.fusion_func == 'concat':
            inp = torch.cat([inp,pe_s,pe_e],dim=-1)
            output = self.proj(inp)
        elif self.fusion_func == 'add':
            output = pe_s + pe_e + inp
        elif self.fusion_func == 'nonlinear_concat':
            pos = self.pos_proj(torch.cat([pe_s,pe_e],dim=-1))
            output = self.proj(torch.cat([inp,pos],dim=-1))
        elif self.fusion_func == 'nonlinear_add':
            pos = self.pos_proj(torch.cat([pe_s,pe_e],dim=-1))
            output = pos + inp
        elif self.fusion_func == 'add_nonlinear':
            inp = inp + pe_s + pe_e
            output = self.proj(inp)

        elif self.fusion_func == 'concat_nonlinear':
            output = self.proj(torch.cat([inp,pe_s,pe_e],dim=-1))



        return output

        # if self.fusion_func == 'add':
        #     result =

    def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        rel pos init:
        如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
        如果是1，那么就按-max_len,max_len来初始化
        """
        num_embeddings = 2 * max_seq_len + 1
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        if rel_pos_init == 0:
            emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        else:
            emb = torch.arange(-max_seq_len, max_seq_len + 1, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

class Absolute_Position_Embedding(nn.Module):
    def __init__(self,fusion_func,hidden_size,learnable,mode=collections.defaultdict(bool),pos_norm=False,max_len=5000):
        '''

        :param hidden_size:
        :param max_len:
        :param learnable:
        :param debug:
        :param fusion_func:暂时只有add和concat(直接拼接然后接线性变换)，后续得考虑直接拼接再接非线性变换
        '''
        super().__init__()
        self.fusion_func = fusion_func
        assert ('add' in self.fusion_func) != ('concat' in self.fusion_func)
        if 'add' in self.fusion_func:
            self.fusion_func = 'add'
        else:
            self.fusion_func = 'concat'
        #备注，在SE绝对位置里，会需要nonlinear操作来融合两种位置pos，但普通的不需要，所以只根据字符串里有哪个关键字来判断
        self.pos_norm = pos_norm
        self.mode = mode
        self.debug = mode['debug']
        self.hidden_size = hidden_size
        pe = get_embedding(max_len,hidden_size)
        # pe = torch.zeros(max_len, hidden_size,requires_grad=True)
        # position = torch.arange(0, max_len).unsqueeze(1).float()
        # div_term = torch.exp(torch.arange(0, hidden_size, 2,dtype=torch.float32) *
        #                      -(math.log(10000.0) / float(hidden_size)))
        # pe[:, 0::2] = torch.sin((position * div_term))
        # pe[:, 1::2] = torch.cos(position * div_term)
        pe_sum = pe.sum(dim=-1,keepdim=True)
        if self.pos_norm:
            with torch.no_grad():
                pe = pe / pe_sum
        pe = pe.unsqueeze(0)
        self.pe = nn.Parameter(pe, requires_grad=learnable)

        if self.fusion_func == 'concat':
            self.proj = nn.Linear(self.hidden_size * 2,self.hidden_size)

        if self.mode['debug']:
            print_info('position embedding:')
            print_info(self.pe[:100])
            print_info('pe size:{}'.format(self.pe.size()))
            print_info('pe avg:{}'.format(torch.sum(self.pe)/(self.pe.size(2)*self.pe.size(1))))
    def forward(self,inp):
        batch = inp.size(0)
        if self.mode['debug']:
            print_info('now in Absolute Position Embedding')
        if self.fusion_func == 'add':
            output = inp + self.pe[:,:inp.size(1)]
        elif self.fusion_func == 'concat':
            inp = torch.cat([inp,self.pe[:,:inp.size(1)].repeat([batch]+[1]*(inp.dim()-1))],dim=-1)
            output = self.proj(inp)

        return output

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb



class Lattice_Transformer_SeqLabel(nn.Module):
    def __init__(self,lattice_embed, bigram_embed, hidden_size, label_size,
                 num_heads, num_layers,
                 use_abs_pos,use_rel_pos, learnable_position,add_position,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 ff_size=-1, scaled=True , dropout=None,use_bigram=True,mode=collections.defaultdict(bool),
                 dvc=None,vocabs=None,
                 rel_pos_shared=True,max_seq_len=-1,k_proj=True,q_proj=True,v_proj=True,r_proj=True,
                 self_supervised=False,attn_ff=True,pos_norm=False,ff_activate='relu',rel_pos_init=0,
                 abs_pos_fusion_func='concat',embed_dropout_pos='0',
                 four_pos_shared=True,four_pos_fusion=None,four_pos_fusion_shared=True,
                 bert_embedding=None,use_pytorch_dropout=False):
        '''
        :param rel_pos_init: 如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
        如果是1，那么就按-max_len,max_len来初始化

        :param embed_dropout_pos: 如果是0，就直接在embed后dropout，是1就在embed变成hidden size之后再dropout，
        是2就在绝对位置加上之后dropout
        '''
        super().__init__()
        self.use_pytorch_dropout = use_pytorch_dropout
        self.four_pos_fusion_shared = four_pos_fusion_shared
        self.mode = mode
        self.four_pos_shared = four_pos_shared
        self.abs_pos_fusion_func = abs_pos_fusion_func
        self.lattice_embed = lattice_embed
        self.bigram_embed = bigram_embed
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        # self.relative_position = relative_position
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert four_pos_fusion is not None
        self.four_pos_fusion = four_pos_fusion
        self.learnable_position = learnable_position
        self.add_position = add_position
        self.rel_pos_shared = rel_pos_shared
        self.self_supervised=self_supervised
        self.vocabs = vocabs
        self.attn_ff = attn_ff
        self.pos_norm = pos_norm
        self.ff_activate = ff_activate
        self.rel_pos_init = rel_pos_init
        self.embed_dropout_pos = embed_dropout_pos


        # if self.relative_position:
        #     print('现在还不支持相对编码！')
        #     exit(1208)

        # if self.add_position:
        #     print('暂时只支持位置编码的concat模式')
        #     exit(1208)

        if self.use_rel_pos and max_seq_len < 0:
            print_info('max_seq_len should be set if relative position encode')
            exit(1208)

        self.max_seq_len = max_seq_len

        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = r_proj

        self.pe = None

        if self.use_abs_pos:
            self.abs_pos_encode = Absolute_SE_Position_Embedding(self.abs_pos_fusion_func,
                                        self.hidden_size,learnable=self.learnable_position,mode=self.mode,
                                        pos_norm=self.pos_norm)

        if self.use_rel_pos:
            pe = get_embedding(max_seq_len,hidden_size,rel_pos_init=self.rel_pos_init)
            pe_sum = pe.sum(dim=-1,keepdim=True)
            if self.pos_norm:
                with torch.no_grad():
                    pe = pe/pe_sum
            self.pe = nn.Parameter(pe, requires_grad=self.learnable_position)
            if self.four_pos_shared:
                self.pe_ss = self.pe
                self.pe_se = self.pe
                self.pe_es = self.pe
                self.pe_ee = self.pe
            else:
                self.pe_ss = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
                self.pe_se = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
                self.pe_es = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
                self.pe_ee = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
        else:
            self.pe = None
            self.pe_ss = None
            self.pe_se = None
            self.pe_es = None
            self.pe_ee = None






        # if self.add_position:
        #     print('现在还不支持位置编码通过concat的方式加入')
        #     exit(1208)

        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        if ff_size==-1:
            ff_size = self.hidden_size
        self.ff_size = ff_size
        self.scaled = scaled
        if dvc == None:
            dvc = 'cpu'
        self.dvc = torch.device(dvc)
        if dropout is None:
            self.dropout = collections.defaultdict(int)
        else:
            self.dropout = dropout
        self.use_bigram = use_bigram

        if self.use_bigram:
            self.bigram_size = self.bigram_embed.embedding.weight.size(1)
            self.char_input_size = self.lattice_embed.embedding.weight.size(1)+self.bigram_embed.embedding.weight.size(1)
        else:
            self.char_input_size = self.lattice_embed.embedding.weight.size(1)

        self.lex_input_size = self.lattice_embed.embedding.weight.size(1)

        if use_pytorch_dropout:
            self.embed_dropout = nn.Dropout(self.dropout['embed'])
            self.gaz_dropout = nn.Dropout(self.dropout['gaz'])
            self.output_dropout = nn.Dropout(self.dropout['output'])
        else:
            self.embed_dropout = MyDropout(self.dropout['embed'])
            self.gaz_dropout = MyDropout(self.dropout['gaz'])
            self.output_dropout = MyDropout(self.dropout['output'])


        self.char_proj = nn.Linear(self.char_input_size,self.hidden_size)
        self.lex_proj = nn.Linear(self.lex_input_size,self.hidden_size)

        self.encoder = Transformer_Encoder(self.hidden_size,self.num_heads,self.num_layers,
                                           relative_position=self.use_rel_pos,
                                           learnable_position=self.learnable_position,
                                           add_position=self.add_position,
                                           layer_preprocess_sequence=self.layer_preprocess_sequence,
                                           layer_postprocess_sequence=self.layer_postprocess_sequence,
                                           dropout=self.dropout,
                                           scaled=self.scaled,
                                           ff_size=self.ff_size,
                                           mode=self.mode,
                                           dvc=self.dvc,
                                           max_seq_len=self.max_seq_len,
                                           pe=self.pe,
                                           pe_ss=self.pe_ss,
                                           pe_se=self.pe_se,
                                           pe_es=self.pe_es,
                                           pe_ee=self.pe_ee,
                                           k_proj=self.k_proj,
                                           q_proj=self.q_proj,
                                           v_proj=self.v_proj,
                                           r_proj=self.r_proj,
                                           attn_ff=self.attn_ff,
                                           ff_activate=self.ff_activate,
                                           lattice=True,
                                           four_pos_fusion=self.four_pos_fusion,
                                           four_pos_fusion_shared=self.four_pos_fusion_shared,
                                           use_pytorch_dropout=self.use_pytorch_dropout)


        self.output = nn.Linear(self.hidden_size,self.label_size)
        if self.self_supervised:
            self.output_self_supervised = nn.Linear(self.hidden_size,len(vocabs['char']))
            print('self.output_self_supervised:{}'.format(self.output_self_supervised.weight.size()))
        self.crf = get_crf_zero_init(self.label_size)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)
        self.batch_num = 0


    def forward(self, lattice, bigrams, seq_len, lex_num, pos_s, pos_e,
                target, chars_target=None):
        # if self.training:
        #     self.batch_num+=1
        # if self.batch_num == 1000:
        #     exit()

        # print('lattice:')
        # print(lattice)
        if self.mode['debug']:
            print('lattice:{}'.format(lattice))
            print('bigrams:{}'.format(bigrams))
            print('seq_len:{}'.format(seq_len))
            print('lex_num:{}'.format(lex_num))
            print('pos_s:{}'.format(pos_s))
            print('pos_e:{}'.format(pos_e))

        batch_size = lattice.size(0)
        max_seq_len_and_lex_num = lattice.size(1)
        max_seq_len = bigrams.size(1)

        raw_embed = self.lattice_embed(lattice)
        #raw_embed 是字和词的pretrain的embedding，但是是分别trian的，所以需要区分对待
        if self.use_bigram:
            bigrams_embed = self.bigram_embed(bigrams)
            bigrams_embed = torch.cat([bigrams_embed,
                                       torch.zeros(size=[batch_size,max_seq_len_and_lex_num-max_seq_len,
                                                         self.bigram_size]).to(bigrams_embed)],dim=1)
            raw_embed_char = torch.cat([raw_embed, bigrams_embed],dim=-1)
        else:
            raw_embed_char = raw_embed

        dim2 = 0
        dim3 = 2
        # print('raw_embed:{}'.format(raw_embed[:,dim2,:dim3]))
        # print('raw_embed_char:{}'.format(raw_embed_char[:, dim2, :dim3]))
        if self.embed_dropout_pos == '0':
            raw_embed_char = self.embed_dropout(raw_embed_char)
            raw_embed = self.gaz_dropout(raw_embed)
        # print('raw_embed_dropout:{}'.format(raw_embed[:,dim2,:dim3]))
        # print('raw_embed_char_dropout:{}'.format(raw_embed_char[:, dim2, :dim3]))

        embed_char = self.char_proj(raw_embed_char)
        if self.mode['debug']:
            print('embed_char:{}'.format(embed_char[:2]))
        char_mask = seq_len_to_mask(seq_len,max_len=max_seq_len_and_lex_num).bool()
        # if self.embed_dropout_pos == '1':
        #     embed_char = self.embed_dropout(embed_char)
        embed_char.masked_fill_(~(char_mask.unsqueeze(-1)), 0)

        embed_lex = self.lex_proj(raw_embed)
        if self.mode['debug']:
            print('embed_lex:{}'.format(embed_lex[:2]))
        # if self.embed_dropout_pos == '1':
        #     embed_lex = self.embed_dropout(embed_lex)

        lex_mask = (seq_len_to_mask(seq_len+lex_num).bool() ^ char_mask.bool())
        embed_lex.masked_fill_(~(lex_mask).unsqueeze(-1), 0)

        assert char_mask.size(1) == lex_mask.size(1)

        embedding = embed_char + embed_lex
        if self.mode['debug']:
            print('embedding:{}'.format(embedding[:2]))

        if self.embed_dropout_pos == '1':
            embedding = self.embed_dropout(embedding)

        if self.use_abs_pos:
            embedding = self.abs_pos_encode(embedding,pos_s,pos_e)

        if self.embed_dropout_pos == '2':
            embedding = self.embed_dropout(embedding)
        # embedding = self.embed_dropout(embedding)

        # print('embedding:{}'.format(embedding[:,dim2,:dim3]))

        if self.batch_num == 327:
            print('{} embed:{}'.format(self.batch_num,
                                       embedding[:2,dim2,:dim3]))

        encoded = self.encoder(embedding,seq_len,lex_num=lex_num,pos_s=pos_s,pos_e=pos_e,
                               print_=(self.batch_num==327))

        if self.batch_num == 327:
            print('{} encoded:{}'.format(self.batch_num,
                                       encoded[:2,dim2,:dim3]))

        if hasattr(self,'output_dropout'):
            encoded = self.output_dropout(encoded)


        encoded = encoded[:,:max_seq_len,:]
        pred = self.output(encoded)

        if self.batch_num == 327:
            print('{} pred:{}'.format(self.batch_num,
                                       pred[:2,dim2,:dim3]))

        # print('pred:{}'.format(pred[:,dim2,:dim3]))
        # exit()

        mask = seq_len_to_mask(seq_len).bool()

        if self.mode['debug']:
            print('debug mode:finish!')
            exit(1208)
        if self.training:
            loss = self.crf(pred, target, mask).mean(dim=0)
            if self.self_supervised:
                # print('self supervised loss added!')
                chars_pred = self.output_self_supervised(encoded)
                chars_pred = chars_pred.view(size=[batch_size*max_seq_len,-1])
                chars_target = chars_target.view(size=[batch_size*max_seq_len])
                self_supervised_loss = self.loss_func(chars_pred,chars_target)
                # print('self_supervised_loss:{}'.format(self_supervised_loss))
                # print('supervised_loss:{}'.format(loss))
                loss += self_supervised_loss

            if self.batch_num == 327:
                print('{} loss:{}'.format(self.batch_num,loss))
                exit()

            # exit()
            return {'loss': loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, mask)
            result = {'pred': pred}
            if self.self_supervised:
                chars_pred = self.output_self_supervised(encoded)
                result['chars_pred'] = chars_pred

            return result


    # def train(self,mode=True):
    #     print('model mode get train ! mode:{}'.format(mode))
    #     super().train(mode)
    #
    # def eval(self):
    #     print('model mode get eval !')
    #     super().eval()




class Transformer_SeqLabel(nn.Module):
    def __init__(self,char_embed, bigram_embed, hidden_size, label_size,
                 num_heads, num_layers,
                 use_abs_pos, use_rel_pos, learnable_position,add_position,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 ff_size=-1, scaled=True , dropout=None,use_bigram=True,mode=collections.defaultdict(bool),
                 dvc=None,vocabs=None,
                 rel_pos_shared=True,max_seq_len=-1,k_proj=True,q_proj=True,v_proj=True,r_proj=True,
                 self_supervised=False,attn_ff=True,pos_norm=False,ff_activate='relu',rel_pos_init=0,
                 abs_pos_fusion_func='concat',embed_dropout_pos='0',):
        '''
        :param rel_pos_init: 如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
        如果是1，那么就按-max_len,max_len来初始化
        '''
        super().__init__()
        self.abs_pos_fusion_func = abs_pos_fusion_func
        self.embed_dropout_pos = embed_dropout_pos
        self.mode = mode
        self.char_embed =char_embed
        self.bigram_embed = bigram_embed
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        # self.relative_position = relative_position
        self.learnable_position = learnable_position
        self.add_position = add_position
        self.rel_pos_shared = rel_pos_shared
        self.self_supervised=self_supervised
        self.vocabs = vocabs
        self.attn_ff = attn_ff
        self.pos_norm = pos_norm
        self.ff_activate = ff_activate
        self.rel_pos_init = rel_pos_init

        if self.use_rel_pos and max_seq_len < 0:
            print_info('max_seq_len should be set if relative position encode')
            exit(1208)

        self.max_seq_len = max_seq_len

        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = r_proj

        self.pe = None
        if self.use_abs_pos:
            self.pos_encode = Absolute_Position_Embedding(self.abs_pos_fusion_func,
                                                          self.hidden_size,learnable=self.learnable_position,mode=self.mode,
                                                          pos_norm=self.pos_norm)


        if self.use_rel_pos:
            pe = get_embedding(max_seq_len,hidden_size,rel_pos_init=self.rel_pos_init)
            pe_sum = pe.sum(dim=-1,keepdim=True)
            if self.pos_norm:
                with torch.no_grad():
                    pe = pe/pe_sum
            self.pe = nn.Parameter(pe, requires_grad=self.learnable_position)
        else:
            self.pe = None


        # if self.relative_position:
        #     print('现在还不支持相对编码！')
        #     exit(1208)




        # if not self.add_position:
        #     print('现在还不支持位置编码通过concat的方式加入')
        #     exit(1208)

        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        if ff_size==-1:
            ff_size = self.hidden_size
        self.ff_size = ff_size
        self.scaled = scaled
        if dvc == None:
            dvc = 'cpu'
        self.dvc = torch.device(dvc)
        if dropout is None:
            self.dropout = collections.defaultdict(int)
        else:
            self.dropout = dropout
        self.use_bigram = use_bigram

        if self.use_bigram:
            self.input_size = self.char_embed.embedding.weight.size(1)+self.bigram_embed.embedding.weight.size(1)
        else:
            self.input_size = self.char_embed.embedding.weight.size(1)

        self.embed_dropout = nn.Dropout(self.dropout['embed'])
        self.w_proj = nn.Linear(self.input_size,self.hidden_size)
        self.encoder = Transformer_Encoder(self.hidden_size,self.num_heads,self.num_layers,
                                           relative_position=self.use_rel_pos,
                                           learnable_position=self.learnable_position,
                                           add_position=self.add_position,
                                           layer_preprocess_sequence=self.layer_preprocess_sequence,
                                           layer_postprocess_sequence=self.layer_postprocess_sequence,
                                           dropout=self.dropout,
                                           scaled=self.scaled,
                                           ff_size=self.ff_size,
                                           mode=self.mode,
                                           dvc=self.dvc,
                                           max_seq_len=self.max_seq_len,
                                           pe=self.pe,
                                           k_proj=self.k_proj,
                                           q_proj=self.q_proj,
                                           v_proj=self.v_proj,
                                           r_proj=self.r_proj,
                                           attn_ff=self.attn_ff,
                                           ff_activate=self.ff_activate,
                                           )

        self.output_dropout = nn.Dropout(self.dropout['output'])

        self.output = nn.Linear(self.hidden_size,self.label_size)
        if self.self_supervised:
            self.output_self_supervised = nn.Linear(self.hidden_size,len(vocabs['char']))
            print('self.output_self_supervised:{}'.format(self.output_self_supervised.weight.size()))
        self.crf = get_crf_zero_init(self.label_size)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)
    def forward(self, chars, bigrams, seq_len, target, chars_target=None):
        # print('**self.training: {} **'.format(self.training))
        batch_size = chars.size(0)
        max_seq_len = chars.size(1)
        chars_embed = self.char_embed(chars)
        if self.use_bigram:
            bigrams_embed = self.bigram_embed(bigrams)
            embedding = torch.cat([chars_embed,bigrams_embed],dim=-1)
        else:
            embedding = chars_embed
        if self.embed_dropout_pos == '0':
            embedding = self.embed_dropout(embedding)

        embedding = self.w_proj(embedding)
        if self.embed_dropout_pos == '1':
            embedding = self.embed_dropout(embedding)

        if self.use_abs_pos:
            embedding = self.pos_encode(embedding)

        if self.embed_dropout_pos == '2':
            embedding = self.embed_dropout(embedding)


        encoded = self.encoder(embedding,seq_len)

        if hasattr(self,'output_dropout'):
            encoded = self.output_dropout(encoded)

        pred = self.output(encoded)

        mask = seq_len_to_mask(seq_len).bool()

        if self.mode['debug']:
            print('debug mode:finish!')
            exit(1208)
        if self.training:
            loss = self.crf(pred, target, mask).mean(dim=0)
            if self.self_supervised:
                # print('self supervised loss added!')
                chars_pred = self.output_self_supervised(encoded)
                chars_pred = chars_pred.view(size=[batch_size*max_seq_len,-1])
                chars_target = chars_target.view(size=[batch_size*max_seq_len])
                self_supervised_loss = self.loss_func(chars_pred,chars_target)
                # print('self_supervised_loss:{}'.format(self_supervised_loss))
                # print('supervised_loss:{}'.format(loss))
                loss += self_supervised_loss
            return {'loss': loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, mask)
            result = {'pred': pred}
            if self.self_supervised:
                chars_pred = self.output_self_supervised(encoded)
                result['chars_pred'] = chars_pred

            return result


    # def train(self,mode=True):
    #     print('model mode get train ! mode:{}'.format(mode))
    #     super().train(mode)
    #
    # def eval(self):
    #     print('model mode get eval !')
    #     super().eval()