import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import collections
from fastNLP import seq_len_to_mask
from utils import print_info, size2MB,MyDropout
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


class Four_Pos_Fusion_Embedding(nn.Module):
    def __init__(self,pe,four_pos_fusion,pe_ss,pe_se,pe_es,pe_ee,max_seq_len,hidden_size,mode):
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.max_seq_len=max_seq_len
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.pe = pe
        self.four_pos_fusion = four_pos_fusion
        if self.four_pos_fusion == 'ff':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size),
                                                    nn.ReLU(inplace=True))
        if self.four_pos_fusion == 'ff_linear':
            self.pos_fusion_forward = nn.Linear(self.hidden_size*4,self.hidden_size)

        elif self.four_pos_fusion == 'ff_two':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*2,self.hidden_size),
                                                    nn.ReLU(inplace=True))
        elif self.four_pos_fusion == 'attn':
            self.w_r = nn.Linear(self.hidden_size,self.hidden_size)
            self.pos_attn_score = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size*4),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size*4,4),
                                                nn.Softmax(dim=-1))

            # print('暂时不支持以attn融合pos信息')
        elif self.four_pos_fusion == 'gate':
            self.w_r = nn.Linear(self.hidden_size,self.hidden_size)
            self.pos_gate_score = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size*2),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size*2,4*self.hidden_size))

            # print('暂时不支持以gate融合pos信息')
            # exit(1208)
    def forward(self,pos_s,pos_e):
        batch = pos_s.size(0)
        #这里的seq_len已经是之前的seq_len+lex_num了
        pos_ss = pos_s.unsqueeze(-1)-pos_s.unsqueeze(-2)
        pos_se = pos_s.unsqueeze(-1)-pos_e.unsqueeze(-2)
        pos_es = pos_e.unsqueeze(-1)-pos_s.unsqueeze(-2)
        pos_ee = pos_e.unsqueeze(-1)-pos_e.unsqueeze(-2)

        if self.mode['debug']:
            print('pos_s:{}'.format(pos_s))
            print('pos_e:{}'.format(pos_e))
            print('pos_ss:{}'.format(pos_ss))
            print('pos_se:{}'.format(pos_se))
            print('pos_es:{}'.format(pos_es))
            print('pos_ee:{}'.format(pos_ee))
        # B prepare relative position encoding
        max_seq_len = pos_s.size(1)
        # rel_distance = self.seq_len_to_rel_distance(max_seq_len)

        # rel_distance_flat = rel_distance.view(-1)
        # rel_pos_embedding_flat = self.pe[rel_distance_flat+self.max_seq_len]
        # rel_pos_embedding = rel_pos_embedding_flat.view(size=[max_seq_len,max_seq_len,self.hidden_size])
        pe_ss = self.pe_ss[(pos_ss).view(-1)+self.max_seq_len].view(size=[batch,max_seq_len,max_seq_len,-1])
        pe_se = self.pe_se[(pos_se).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_es = self.pe_es[(pos_es).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_ee = self.pe_ee[(pos_ee).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])

        # print('pe_ss:{}'.format(pe_ss.size()))

        if self.four_pos_fusion == 'ff':
            pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
            if self.mode['gpumm']:
                print('四个位置合起来:{},{}'.format(pe_4.size(),size2MB(pe_4.size())))
            rel_pos_embedding = self.pos_fusion_forward(pe_4)
        if self.four_pos_fusion == 'ff_linear':
            pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
            if self.mode['gpumm']:
                print('四个位置合起来:{},{}'.format(pe_4.size(),size2MB(pe_4.size())))
            rel_pos_embedding = self.pos_fusion_forward(pe_4)
        if self.four_pos_fusion == 'ff_two':
            pe_2 = torch.cat([pe_ss,pe_ee],dim=-1)
            if self.mode['gpumm']:
                print('2个位置合起来:{},{}'.format(pe_2.size(),size2MB(pe_2.size())))
            rel_pos_embedding = self.pos_fusion_forward(pe_2)
        elif self.four_pos_fusion == 'attn':
            pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
            attn_score = self.pos_attn_score(pe_4)
            pe_4_unflat = self.w_r(pe_4.view(batch,max_seq_len,max_seq_len,4,self.hidden_size))
            pe_4_fusion = (attn_score.unsqueeze(-1) * pe_4_unflat).sum(dim=-2)
            rel_pos_embedding = pe_4_fusion
            if self.mode['debug']:
                print('pe_4照理说应该是 Batch * SeqLen * SeqLen * HiddenSize')
                print(pe_4_fusion.size())

        elif self.four_pos_fusion == 'gate':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            gate_score = self.pos_gate_score(pe_4).view(batch,max_seq_len,max_seq_len,4,self.hidden_size)
            gate_score = F.softmax(gate_score,dim=-2)
            pe_4_unflat = self.w_r(pe_4.view(batch, max_seq_len, max_seq_len, 4, self.hidden_size))
            pe_4_fusion = (gate_score * pe_4_unflat).sum(dim=-2)
            rel_pos_embedding = pe_4_fusion


        return rel_pos_embedding


class MultiHead_Attention_Lattice_rel_save_gpumm(nn.Module):
    def __init__(self, hidden_size, num_heads, pe,
                 pe_ss,pe_se,pe_es,pe_ee,
                 scaled=True, max_seq_len=-1,
                 dvc=None,mode=collections.defaultdict(bool),k_proj=True,q_proj=True,v_proj=True,r_proj=True,
                 attn_dropout=None,
                 ff_final=True,
                 four_pos_fusion=None):
        '''

        :param hidden_size:
        :param num_heads:
        :param scaled:
        :param debug:
        :param max_seq_len:
        :param device:
        '''
        super().__init__()
        assert four_pos_fusion is not None
        self.four_pos_fusion = four_pos_fusion
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.mode = mode
        if self.mode['debug']:
            print_info('rel pos attn')
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        self.max_seq_len = max_seq_len
        if dvc is None:
            dvc = torch.device('cpu')
        self.dvc = dvc
        assert (self.per_head_size * self.num_heads == self.hidden_size)

        self.k_proj=k_proj
        self.q_proj=q_proj
        self.v_proj=v_proj
        self.r_proj=r_proj

        if self.four_pos_fusion == 'ff':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size),
                                                    nn.ReLU(inplace=True))
        elif self.four_pos_fusion == 'attn':
            self.pos_attn_score = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size*4),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size*4,4),
                                                nn.Softmax(dim=-1))

            # print('暂时不支持以attn融合pos信息')
        elif self.four_pos_fusion == 'gate':
            self.pos_gate_score = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size*2),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size*2,4*self.hidden_size))

            # print('暂时不支持以gate融合pos信息')
            # exit(1208)


        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.Tensor(self.num_heads,self.per_head_size))
        self.v = nn.Parameter(torch.Tensor(self.num_heads,self.per_head_size))

        self.pe = pe

        self.dropout = MyDropout(attn_dropout)

        if ff_final:
            self.ff_final = nn.Linear(self.hidden_size,self.hidden_size)



    def forward(self,key, query, value, seq_len, lex_num, pos_s,pos_e,rel_pos_embedding):
        batch = key.size(0)
        #这里的seq_len已经是之前的seq_len+lex_num了

        #开始计算相对位置融合

        # pos_ss = pos_s.unsqueeze(-1)-pos_s.unsqueeze(-2)
        # pos_se = pos_s.unsqueeze(-1)-pos_e.unsqueeze(-2)
        # pos_es = pos_e.unsqueeze(-1)-pos_s.unsqueeze(-2)
        # pos_ee = pos_e.unsqueeze(-1)-pos_e.unsqueeze(-2)
        #
        # if self.mode['debug']:
        #     print('pos_s:{}'.format(pos_s))
        #     print('pos_e:{}'.format(pos_e))
        #     print('pos_ss:{}'.format(pos_ss))
        #     print('pos_se:{}'.format(pos_se))
        #     print('pos_es:{}'.format(pos_es))
        #     print('pos_ee:{}'.format(pos_ee))
        # # B prepare relative position encoding
        # max_seq_len = key.size(1)
        # # rel_distance = self.seq_len_to_rel_distance(max_seq_len)
        #
        # # rel_distance_flat = rel_distance.view(-1)
        # # rel_pos_embedding_flat = self.pe[rel_distance_flat+self.max_seq_len]
        # # rel_pos_embedding = rel_pos_embedding_flat.view(size=[max_seq_len,max_seq_len,self.hidden_size])
        # pe_ss = self.pe[(pos_ss).view(-1)+self.max_seq_len].view(size=[batch,max_seq_len,max_seq_len,-1])
        # pe_se = self.pe[(pos_se).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        # pe_es = self.pe[(pos_es).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        # pe_ee = self.pe[(pos_ee).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        #
        # # print('pe_ss:{}'.format(pe_ss.size()))
        #
        # if self.four_pos_fusion == 'ff':
        #     pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
        #     if self.mode['gpumm']:
        #         print('四个位置合起来:{},{}'.format(pe_4.size(),size2MB(pe_4.size())))
        #     rel_pos_embedding = self.pos_fusion_forward(pe_4)
        # elif self.four_pos_fusion == 'attn':
        #     pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
        #     attn_score = self.pos_attn_score(pe_4)
        #     pe_4_unflat = pe_4.view(batch,max_seq_len,max_seq_len,4,self.hidden_size)
        #     pe_4_fusion = (attn_score.unsqueeze(-1) * pe_4_unflat).sum(dim=-2)
        #     rel_pos_embedding = pe_4_fusion
        #     if self.mode['debug']:
        #         print('pe_4照理说应该是 Batch * SeqLen * SeqLen * HiddenSize')
        #         print(pe_4_fusion.size())
        # elif self.four_pos_fusion == 'gate':
        #     pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
        #     gate_score = self.pos_gate_score(pe_4).view(batch,max_seq_len,max_seq_len,4,self.hidden_size)
        #     gate_score = F.softmax(gate_score,dim=-2)
        #     pe_4_unflat = pe_4.view(batch, max_seq_len, max_seq_len, 4, self.hidden_size)
        #     pe_4_fusion = (gate_score * pe_4_unflat).sum(dim=-2)
        #     rel_pos_embedding = pe_4_fusion
        #
        #结束计算相对位置融合

        # E prepare relative position encoding

        if self.k_proj:
            if self.mode['debug']:
                print_info('k_proj!')
            key = self.w_k(key)
        if self.q_proj:
            if self.mode['debug']:
                print_info('q_proj!')
            query = self.w_q(query)
        if self.v_proj:
            if self.mode['debug']:
                print_info('v_proj!')
            value = self.w_v(value)
        if self.r_proj:
            if self.mode['debug']:
                print_info('r_proj!')
            rel_pos_embedding = self.w_r(rel_pos_embedding)

        batch = key.size(0)
        max_seq_len = key.size(1)


        # batch * seq_len * n_head * d_head
        key = torch.reshape(key, [batch, max_seq_len, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, max_seq_len, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, max_seq_len, self.num_heads, self.per_head_size])
        rel_pos_embedding = torch.reshape(rel_pos_embedding,
                                          [batch,max_seq_len, max_seq_len, self.num_heads,self.per_head_size])


        # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)



        # batch * n_head * d_head * key_len
        key = key.transpose(-1, -2)
        # #A
        # A_ = torch.matmul(query,key)
        # #C
        # # key: batch * n_head * d_head * key_len
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        # u_for_c: 1(batch broadcast) * num_heads * 1 *per_head_size
        # key_for_c = key
        # C_ = torch.matmul(u_for_c, key)
        query_and_u_for_c = query + u_for_c
        if self.mode['debug']:
            print('query:{}'.format(query.size()))
            print('u_for_c:{}'.format(u_for_c.size()))
            print('query_and_u_for_c:{}'.format(query_and_u_for_c.size()))
            print('key:{}'.format(key.size()))
        A_C = torch.matmul(query_and_u_for_c, key)

        if self.mode['debug']:
            print('query size:{}'.format(query.size()))
            print('query_and_u_for_c size:{}'.format(query_and_u_for_c.size()))

        #B
        rel_pos_embedding_for_b = rel_pos_embedding.permute(0, 3, 1, 4, 2)
        # after above, rel_pos_embedding: batch * num_head * query_len * per_head_size * key_len
        query_for_b = query.view([batch, self.num_heads, max_seq_len, 1, self.per_head_size])
        # after above, query_for_b: batch * num_head * query_len * 1 * per_head_size
        # print('query for b:{}'.format(query_for_b.size()))
        # print('rel_pos_embedding_for_b{}'.format(rel_pos_embedding_for_b.size()))
        # B_ = torch.matmul(query_for_b,rel_pos_embedding_for_b).squeeze(-2)

        #D
        # rel_pos_embedding_for_d = rel_pos_embedding.unsqueeze(-2)
        # after above, rel_pos_embedding: batch * query_seq_len * key_seq_len * num_heads * 1 *per_head_size
        # v_for_d = self.v.unsqueeze(-1)
        # v_for_d: num_heads * per_head_size * 1
        # D_ = torch.matmul(rel_pos_embedding_for_d,v_for_d).squeeze(-1).squeeze(-1).permute(0,3,1,2)

        query_for_b_and_v_for_d = query_for_b + self.v.view(1,self.num_heads,1,1,self.per_head_size)
        B_D = torch.matmul(query_for_b_and_v_for_d, rel_pos_embedding_for_b).squeeze(-2)
        #att_score: Batch * num_heads * query_len * key_len
        # A, B C and D is exactly the shape
        if self.mode['debug']:
            print_info('AC:{}'.format(A_C.size()))
            print_info('BD:{}'.format(B_D.size()))
            # print_info('A:{}'.format(A_.size()))
            # print_info('B:{}'.format(B_.size()))
            # print_info('C:{}'.format(C_.size()))
            # print_info('D:{}'.format(D_.size()))
        attn_score_raw = A_C + B_D

        if self.scaled:
            attn_score_raw  = attn_score_raw / math.sqrt(self.per_head_size)

        mask = seq_len_to_mask(seq_len+lex_num).bool().unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attn_score_raw.masked_fill(~mask, -1e15)
        if self.mode['debug']:
            print('attn_score_raw_masked:{}'.format(attn_score_raw_masked))
            print('seq_len:{}'.format(seq_len))

        attn_score = F.softmax(attn_score_raw_masked,dim=-1)

        attn_score = self.dropout(attn_score)

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1,2).contiguous(). \
            reshape(batch, max_seq_len, self.hidden_size)


        if hasattr(self,'ff_final'):
            print('ff_final!!')
            result = self.ff_final(result)

        return result

    def seq_len_to_rel_distance(self,max_seq_len):
        '''

        :param seq_len: seq_len batch
        :return: L*L rel_distance
        '''
        index = torch.arange(0, max_seq_len)
        assert index.size(0) == max_seq_len
        assert index.dim() == 1
        index = index.repeat(max_seq_len, 1)
        offset = torch.arange(0, max_seq_len).unsqueeze(1)
        offset = offset.repeat(1, max_seq_len)
        index = index - offset
        index = index.to(self.dvc)
        return index

class MultiHead_Attention_Lattice_rel(nn.Module):
    def __init__(self, hidden_size, num_heads, pe,
                 pe_ss,pe_se,pe_es,pe_ee,
                 scaled=True, max_seq_len=-1,
                 dvc=None,mode=collections.defaultdict(bool),k_proj=True,q_proj=True,v_proj=True,r_proj=True,
                 attn_dropout=None,
                 ff_final=True,
                 four_pos_fusion=None):
        '''

        :param hidden_size:
        :param num_heads:
        :param scaled:
        :param debug:
        :param max_seq_len:
        :param device:
        '''
        super().__init__()
        assert four_pos_fusion is not None
        self.four_pos_fusion = four_pos_fusion
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.mode = mode
        if self.mode['debug']:
            print_info('rel pos attn')
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        self.max_seq_len = max_seq_len
        if dvc is None:
            dvc = torch.device('cpu')
        self.dvc = dvc
        assert (self.per_head_size * self.num_heads == self.hidden_size)

        self.k_proj=k_proj
        self.q_proj=q_proj
        self.v_proj=v_proj
        self.r_proj=r_proj

        if self.four_pos_fusion == 'ff':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size),
                                                    nn.ReLU(inplace=True))
        elif self.four_pos_fusion == 'attn':
            self.pos_attn_score = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size*4),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size*4,4),
                                                nn.Softmax(dim=-1))

            # print('暂时不支持以attn融合pos信息')
        elif self.four_pos_fusion == 'gate':
            self.pos_attn_score = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size*2),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size*2,4),
                                                nn.Softmax(dim=-1))
            print('暂时不支持以gate融合pos信息')
            exit(1208)


        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.Tensor(self.num_heads,self.per_head_size))
        self.v = nn.Parameter(torch.Tensor(self.num_heads,self.per_head_size))

        self.pe = pe

        self.dropout = MyDropout(attn_dropout)

        if ff_final:
            self.ff_final = nn.Linear(self.hidden_size,self.hidden_size)



    def forward(self,key, query, value, seq_len, lex_num, pos_s,pos_e):
        batch = key.size(0)
        #这里的seq_len已经是之前的seq_len+lex_num了
        pos_ss = pos_s.unsqueeze(-1)-pos_s.unsqueeze(-2)
        pos_se = pos_s.unsqueeze(-1)-pos_e.unsqueeze(-2)
        pos_es = pos_e.unsqueeze(-1)-pos_s.unsqueeze(-2)
        pos_ee = pos_e.unsqueeze(-1)-pos_e.unsqueeze(-2)

        if self.mode['debug']:
            print('pos_s:{}'.format(pos_s))
            print('pos_e:{}'.format(pos_e))
            print('pos_ss:{}'.format(pos_ss))
            print('pos_se:{}'.format(pos_se))
            print('pos_es:{}'.format(pos_es))
            print('pos_ee:{}'.format(pos_ee))
        # B prepare relative position encoding
        max_seq_len = key.size(1)
        # rel_distance = self.seq_len_to_rel_distance(max_seq_len)

        # rel_distance_flat = rel_distance.view(-1)
        # rel_pos_embedding_flat = self.pe[rel_distance_flat+self.max_seq_len]
        # rel_pos_embedding = rel_pos_embedding_flat.view(size=[max_seq_len,max_seq_len,self.hidden_size])
        pe_ss = self.pe[(pos_ss).view(-1)+self.max_seq_len].view(size=[batch,max_seq_len,max_seq_len,-1])
        pe_se = self.pe[(pos_se).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_es = self.pe[(pos_es).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_ee = self.pe[(pos_ee).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])

        # print('pe_ss:{}'.format(pe_ss.size()))

        if self.four_pos_fusion == 'ff':
            pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
            if self.mode['gpumm']:
                print('四个位置合起来:{},{}'.format(pe_4.size(),size2MB(pe_4.size())))
            rel_pos_embedding = self.pos_fusion_forward(pe_4)
        elif self.four_pos_fusion == 'attn':
            pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
            attn_score = self.pos_attn_score(pe_4)
            pe_4_unflat = pe_4.view(batch,max_seq_len,max_seq_len,4,self.hidden_size)
            pe_4_fusion = (attn_score.unsqueeze(-1) * pe_4_unflat).sum(-2)
            rel_pos_embedding = pe_4_fusion
            if self.mode['debug']:
                print('pe_4照理说应该是 Batch * SeqLen * SeqLen * HiddenSize')
                print(pe_4_fusion.size())


        # E prepare relative position encoding

        if self.k_proj:
            if self.mode['debug']:
                print_info('k_proj!')
            key = self.w_k(key)
        if self.q_proj:
            if self.mode['debug']:
                print_info('q_proj!')
            query = self.w_q(query)
        if self.v_proj:
            if self.mode['debug']:
                print_info('v_proj!')
            value = self.w_v(value)
        if self.r_proj:
            if self.mode['debug']:
                print_info('r_proj!')
            rel_pos_embedding = self.w_r(rel_pos_embedding)

        batch = key.size(0)
        max_seq_len = key.size(1)


        # batch * seq_len * n_head * d_head
        key = torch.reshape(key, [batch, max_seq_len, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, max_seq_len, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, max_seq_len, self.num_heads, self.per_head_size])
        rel_pos_embedding = torch.reshape(rel_pos_embedding,
                                          [batch,max_seq_len, max_seq_len, self.num_heads,self.per_head_size])


        # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)



        # batch * n_head * d_head * key_len
        key = key.transpose(-1, -2)


        #A
        A_ = torch.matmul(query,key)

        #B
        rel_pos_embedding_for_b = rel_pos_embedding.permute(0, 3, 1, 4, 2)
        # after above, rel_pos_embedding: batch * num_head * query_len * per_head_size * key_len
        query_for_b = query.view([batch, self.num_heads, max_seq_len, 1, self.per_head_size])
        # print('query for b:{}'.format(query_for_b.size()))
        # print('rel_pos_embedding_for_b{}'.format(rel_pos_embedding_for_b.size()))
        B_ = torch.matmul(query_for_b,rel_pos_embedding_for_b).squeeze(-2)

        #D
        rel_pos_embedding_for_d = rel_pos_embedding.unsqueeze(-2)
        # after above, rel_pos_embedding: batch * query_seq_len * key_seq_len * num_heads * 1 *per_head_size
        v_for_d = self.v.unsqueeze(-1)
        # v_for_d: num_heads * per_head_size * 1
        D_ = torch.matmul(rel_pos_embedding_for_d,v_for_d).squeeze(-1).squeeze(-1).permute(0,3,1,2)

        #C
        # key: batch * n_head * d_head * key_len
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        # u_for_c: 1(batch broadcast) * num_heads * 1 *per_head_size
        key_for_c = key
        C_ = torch.matmul(u_for_c, key)

        #att_score: Batch * num_heads * query_len * key_len
        # A, B C and D is exactly the shape
        if self.mode['debug']:
            print_info('A:{}'.format(A_.size()))
            print_info('B:{}'.format(B_.size()))
            print_info('C:{}'.format(C_.size()))
            print_info('D:{}'.format(D_.size()))
        attn_score_raw = A_ + B_ + C_ + D_

        if self.scaled:
            attn_score_raw  = attn_score_raw / math.sqrt(self.per_head_size)

        mask = seq_len_to_mask(seq_len+lex_num).bool().unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attn_score_raw.masked_fill(~mask, -1e15)
        if self.mode['debug']:
            print('attn_score_raw_masked:{}'.format(attn_score_raw_masked))
            print('seq_len:{}'.format(seq_len))

        attn_score = F.softmax(attn_score_raw_masked,dim=-1)

        attn_score = self.dropout(attn_score)

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1,2).contiguous(). \
            reshape(batch, max_seq_len, self.hidden_size)


        if hasattr(self,'ff_final'):
            print('ff_final!!')
            result = self.ff_final(result)

        return result

    def seq_len_to_rel_distance(self,max_seq_len):
        '''

        :param seq_len: seq_len batch
        :return: L*L rel_distance
        '''
        index = torch.arange(0, max_seq_len)
        assert index.size(0) == max_seq_len
        assert index.dim() == 1
        index = index.repeat(max_seq_len, 1)
        offset = torch.arange(0, max_seq_len).unsqueeze(1)
        offset = offset.repeat(1, max_seq_len)
        index = index - offset
        index = index.to(self.dvc)
        return index

class MultiHead_Attention_rel(nn.Module):
    def __init__(self, hidden_size, num_heads, pe, scaled=True, max_seq_len=-1,
                 dvc=None,mode=collections.defaultdict(bool),k_proj=True,q_proj=True,v_proj=True,r_proj=True,
                 attn_dropout=None,
                 ff_final=True):
        '''

        :param hidden_size:
        :param num_heads:
        :param scaled:
        :param debug:
        :param max_seq_len:
        :param device:
        '''
        super().__init__()
        self.mode=mode
        if self.mode['debug']:
            print_info('rel pos attn')
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        self.max_seq_len = max_seq_len
        if dvc is None:
            dvc = torch.device('cpu')
        self.dvc = dvc
        assert (self.per_head_size * self.num_heads == self.hidden_size)

        self.k_proj=k_proj
        self.q_proj=q_proj
        self.v_proj=v_proj
        self.r_proj=r_proj


        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.Tensor(self.num_heads,self.per_head_size))
        self.v = nn.Parameter(torch.Tensor(self.num_heads,self.per_head_size))

        self.pe = pe

        self.dropout = MyDropout(attn_dropout)

        if ff_final:
            self.ff_final = nn.Linear(self.hidden_size,self.hidden_size)



    def forward(self,key, query, value, seq_len):
        # B prepare relative position encoding
        max_seq_len = torch.max(seq_len)
        rel_distance = self.seq_len_to_rel_distance(max_seq_len)
        rel_distance_flat = rel_distance.view(-1)
        rel_pos_embedding_flat = self.pe[rel_distance_flat+self.max_seq_len]
        rel_pos_embedding = rel_pos_embedding_flat.view(size=[max_seq_len,max_seq_len,self.hidden_size])
        # E prepare relative position encoding

        if self.k_proj:
            if self.mode['debug']:
                print_info('k_proj!')
            key = self.w_k(key)
        if self.q_proj:
            if self.mode['debug']:
                print_info('q_proj!')
            query = self.w_q(query)
        if self.v_proj:
            if self.mode['debug']:
                print_info('v_proj!')
            value = self.w_v(value)
        if self.r_proj:
            if self.mode['debug']:
                print_info('r_proj!')
            rel_pos_embedding = self.w_r(rel_pos_embedding)

        batch = key.size(0)
        max_seq_len = key.size(1)


        # batch * seq_len * n_head * d_head
        key = torch.reshape(key, [batch, max_seq_len, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, max_seq_len, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, max_seq_len, self.num_heads, self.per_head_size])
        rel_pos_embedding = torch.reshape(rel_pos_embedding,
                                          [max_seq_len, max_seq_len, self.num_heads,self.per_head_size])


        # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)



        # batch * n_head * d_head * key_len
        key = key.transpose(-1, -2)


        #A
        A_ = torch.matmul(query,key)

        #B
        rel_pos_embedding_for_b = rel_pos_embedding.unsqueeze(0).permute(0, 3, 1, 4, 2)
        # after above, rel_pos_embedding: batch * num_head * query_len * per_head_size * key_len
        query_for_b = query.view([batch, self.num_heads, max_seq_len, 1, self.per_head_size])
        # print('query for b:{}'.format(query_for_b.size()))
        # print('rel_pos_embedding_for_b{}'.format(rel_pos_embedding_for_b.size()))
        B_ = torch.matmul(query_for_b,rel_pos_embedding_for_b).squeeze(-2)

        #D
        rel_pos_embedding_for_d = rel_pos_embedding.unsqueeze(-2)
        # after above, rel_pos_embedding: query_seq_len * key_seq_len * num_heads * 1 *per_head_size
        v_for_d = self.v.unsqueeze(-1)
        # v_for_d: num_heads * per_head_size * 1
        D_ = torch.matmul(rel_pos_embedding_for_d,v_for_d).squeeze(-1).squeeze(-1).permute(2,0,1).unsqueeze(0)

        #C
        # key: batch * n_head * d_head * key_len
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        # u_for_c: 1(batch broadcast) * num_heads * 1 *per_head_size
        key_for_c = key
        C_ = torch.matmul(u_for_c, key)

        #att_score: Batch * num_heads * query_len * key_len
        # A, B C and D is exactly the shape
        if self.mode['debug']:
            print_info('A:{}'.format(A_.size()))
            print_info('B:{}'.format(B_.size()))
            print_info('C:{}'.format(C_.size()))
            print_info('D:{}'.format(D_.size()))
        attn_score_raw = A_ + B_ + C_ + D_

        if self.scaled:
            attn_score_raw  = attn_score_raw / math.sqrt(self.per_head_size)

        mask = seq_len_to_mask(seq_len).bool().unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attn_score_raw.masked_fill(~mask, -1e15)
        if self.mode['debug']:
            print('attn_score_raw_masked:{}'.format(attn_score_raw_masked))
            print('seq_len:{}'.format(seq_len))

        attn_score = F.softmax(attn_score_raw_masked,dim=-1)

        attn_score = self.dropout(attn_score)

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1,2).contiguous(). \
            reshape(batch, max_seq_len, self.hidden_size)


        if hasattr(self,'ff_final'):
            print('ff_final!!')
            result = self.ff_final(result)

        return result

    def seq_len_to_rel_distance(self,max_seq_len):
        '''

        :param seq_len: seq_len batch
        :return: L*L rel_distance
        '''
        index = torch.arange(0, max_seq_len)
        assert index.size(0) == max_seq_len
        assert index.dim() == 1
        index = index.repeat(max_seq_len, 1)
        offset = torch.arange(0, max_seq_len).unsqueeze(1)
        offset = offset.repeat(1, max_seq_len)
        index = index - offset
        index = index.to(self.dvc)
        return index


class MultiHead_Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, scaled=True,mode=collections.defaultdict(bool), k_proj=True,q_proj=True,v_proj=True,
                 attn_dropout=None,ff_final=True):
        super().__init__()
        #这个模型接受的输入本身是带有位置信息的，适用于transformer的绝对位置编码模式
        # TODO: attention dropout
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        assert (self.per_head_size * self.num_heads == self.hidden_size)
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        if ff_final:
            self.ff_final = nn.Linear(self.hidden_size, self.hidden_size)

        self.mode = mode
        self.k_proj=k_proj
        self.q_proj=q_proj
        self.v_proj=v_proj
        if self.mode['debug']:
            print_info('abs pos attn')

        if attn_dropout == None:
            dropout = collections.defaultdict(int)
        self.dropout = MyDropout(attn_dropout)


    def forward(self, key, query, value, seq_len, lex_num=0):
        if self.k_proj:
            key = self.w_k(key)
        if self.q_proj:
            query = self.w_q(query)
        if self.v_proj:
            value = self.w_v(value)

        batch = key.size(0)
        max_seq_len = key.size(1)

        key = torch.reshape(key, [batch, max_seq_len, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, max_seq_len, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, max_seq_len, self.num_heads, self.per_head_size])

        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        key = key.transpose(-1, -2)

        attention_raw = torch.matmul(query, key)

        if self.scaled:
            attention_raw = attention_raw / math.sqrt(self.per_head_size)

        # if self.mode['debug']:
        #     print('attention_raw:{}'.format(attention_raw.size()))
        #     print('mask:{},{}'.format(mask.size(),mask.dtype))
        #     print('mask==0:{}'.format((mask==0).dtype))
        mask = seq_len_to_mask(seq_len + lex_num).bool().unsqueeze(1).unsqueeze(1)
        attention_raw_masked = attention_raw.masked_fill(~mask, -1e15)

        attn_score = F.softmax(attention_raw_masked, dim=-1)
        attn_score = self.dropout(attn_score)
        # TODO attention dropout

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1, 2).contiguous(). \
            reshape(batch, max_seq_len, self.hidden_size)

        if hasattr(self,'ff_final'):
            result = self.ff_final(result)

        return result

class Positionwise_FeedForward(nn.Module):
    def __init__(self, sizes, dropout=None,ff_activate='relu'):
        super().__init__()
        self.num_layers = len(sizes)-1
        for i in range(self.num_layers):
            setattr(self, 'w' + str(i), nn.Linear(sizes[i], sizes[i + 1]))

        if dropout == None:
            dropout = collections.defaultdict(int)

        self.dropout = MyDropout(dropout['ff'])
        self.dropout_2 = MyDropout(dropout['ff_2'])
        if ff_activate == 'relu':
            self.activate = nn.ReLU(inplace=True)
        elif ff_activate == 'leaky':
            self.activate = nn.LeakyReLU(inplace=True)

    def forward(self, inp):
        output = inp
        for i in range(self.num_layers):
            if i != 0:
                output = self.activate(output)
            w = getattr(self, 'w' + str(i))
            output = w(output)
            if i == 0:
                output = self.dropout(output)
            if i == 1:
                output = self.dropout_2(output)

        return output


class Absolute_Position_Embedding(nn.Module):
    def __init__(self,hidden_size,max_len=5000,learnable=False,mode=collections.defaultdict(bool),pos_norm=False):
        '''

        :param hidden_size:
        :param max_len:
        :param learnable:
        :param debug:
        '''
        super().__init__()
        self.pos_norm = pos_norm
        self.mode=mode
        pe = Absolute_Position_Embedding.get_embedding(max_len,hidden_size)
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
        if self.mode['debug']:
            print_info('position embedding:')
            print_info(self.pe[:100])
            print_info('pe size:{}'.format(self.pe.size()))
            print_info('pe avg:{}'.format(torch.sum(self.pe)/(self.pe.size(2)*self.pe.size(1))))
    def forward(self,inp):
        if self.mode['debug']:
            print_info('now in Absolute Position Embedding')
        return inp + self.pe[:,:inp.size(1)]

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



class Transformer_Encoder_Layer(nn.Module):
    def __init__(self, hidden_size, num_heads,
                 relative_position, learnable_position, add_position,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 dropout=None, scaled=True, ff_size=-1,mode=collections.defaultdict(bool),
                 max_seq_len=-1,pe=None,
                 pe_ss=None, pe_se=None, pe_es=None, pe_ee=None,
                 dvc=None,
                 k_proj=True,q_proj=True,v_proj=True,r_proj=True,
                 attn_ff=True,ff_activate='relu',lattice=False,
                 four_pos_shared=True,four_pos_fusion=None,four_pos_fusion_embedding=None
                 ):
        super().__init__()
        self.four_pos_fusion_embedding=four_pos_fusion_embedding
        self.four_pos_shared=four_pos_shared
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.lattice = lattice
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.relative_position = relative_position
        if self.relative_position and self.lattice:
            assert four_pos_fusion is not None
        self.four_pos_fusion = four_pos_fusion
        self.learnable_position = learnable_position
        self.add_position = add_position
        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        self.scaled = scaled
        self.mode = mode
        self.attn_ff = attn_ff
        self.ff_activate = ff_activate

        if self.relative_position and max_seq_len < 0:
            print_info('max_seq_len should be set if relative position encode')
            exit(1208)

        self.max_seq_len = max_seq_len
        if dvc is None:
            dvc = torch.device('cpu')
        self.dvc = dvc

        self.k_proj=k_proj
        self.q_proj=q_proj
        self.v_proj=v_proj
        self.r_proj=r_proj
        import copy
        if self.relative_position:
            if pe is None:
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
                self.pe = pe
                self.pe_ss = pe_ss
                self.pe_se = pe_se
                self.pe_es = pe_es
                self.pe_ee = pe_ee
        if self.four_pos_fusion_embedding is None:
            self.four_pos_fusion_embedding = \
                Four_Pos_Fusion_Embedding(self.pe,self.four_pos_fusion,self.pe_ss,self.pe_se,self.pe_es,self.pe_ee,
                                          self.max_seq_len,self.hidden_size,self.mode)


        # if self.relative_position:
        #     print('现在还不支持相对编码！')
        #     exit(1208)

        # if not self.add_position:
        #     print_info('现在还不支持位置编码通过concat的方式加入')
        #     exit(1208)

        if dropout == None:
            dropout = collections.defaultdict(int)
        self.dropout = dropout

        if ff_size == -1:
            ff_size = hidden_size
        self.ff_size = ff_size
        # print('dropout:{}'.format(self.dropout))
        self.layer_preprocess = Layer_Process(self.layer_preprocess_sequence,self.hidden_size,self.dropout['pre'])
        self.layer_postprocess = Layer_Process(self.layer_postprocess_sequence,self.hidden_size,self.dropout['post'])
        if self.relative_position:
            if not self.lattice:
                self.attn = MultiHead_Attention_rel(self.hidden_size, self.num_heads,
                                                    pe=self.pe,
                                                    scaled=self.scaled,
                                                    mode=self.mode,
                                                    max_seq_len=self.max_seq_len,
                                                    dvc=self.dvc,
                                                    k_proj=self.k_proj,
                                                    q_proj=self.q_proj,
                                                    v_proj=self.v_proj,
                                                    r_proj=self.r_proj,
                                                    attn_dropout=self.dropout['attn'],
                                                    ff_final=self.attn_ff)
            else:
                self.attn = MultiHead_Attention_Lattice_rel_save_gpumm(self.hidden_size, self.num_heads,
                                                    pe=self.pe,
                                                    pe_ss=self.pe_ss,
                                                    pe_se=self.pe_se,
                                                    pe_es=self.pe_es,
                                                    pe_ee=self.pe_ee,
                                                    scaled=self.scaled,
                                                    mode=self.mode,
                                                    max_seq_len=self.max_seq_len,
                                                    dvc=self.dvc,
                                                    k_proj=self.k_proj,
                                                    q_proj=self.q_proj,
                                                    v_proj=self.v_proj,
                                                    r_proj=self.r_proj,
                                                    attn_dropout=self.dropout['attn'],
                                                    ff_final=self.attn_ff,
                                                    four_pos_fusion=self.four_pos_fusion)

        else:
            self.attn = MultiHead_Attention(self.hidden_size, self.num_heads, self.scaled, mode=self.mode,
                                            k_proj=self.k_proj,q_proj=self.q_proj,v_proj=self.v_proj,
                                            attn_dropout=self.dropout['attn'],
                                            ff_final=self.attn_ff)



        self.ff = Positionwise_FeedForward([hidden_size, ff_size, hidden_size], self.dropout,ff_activate=self.ff_activate)

    def forward(self, inp, seq_len, lex_num=0,pos_s=None,pos_e=None,rel_pos_embedding=None):
        output = inp
        output = self.layer_preprocess(output)
        if self.lattice:
            if self.relative_position:
                if rel_pos_embedding is None:
                    rel_pos_embedding = self.four_pos_fusion_embedding(pos_s,pos_e)
                output = self.attn(output, output, output, seq_len, pos_s=pos_s, pos_e=pos_e, lex_num=lex_num,
                                   rel_pos_embedding=rel_pos_embedding)
            else:
                output = self.attn(output, output, output, seq_len, lex_num)
        else:
            output = self.attn(output, output, output, seq_len)
        output = self.layer_postprocess(output)
        output = self.layer_preprocess(output)
        output = self.ff(output)
        output = self.layer_postprocess(output)

        return output


class Layer_Process(nn.Module):
    def __init__(self, process_sequence, hidden_size, dropout=0, ):
        super().__init__()
        self.process_sequence = process_sequence.lower()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        if 'd' in self.process_sequence:
            self.dropout = MyDropout(dropout)
        if 'n' in self.process_sequence:
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inp):
        output = inp
        for op in self.process_sequence:
            if op == 'a':
                output = output + inp
            elif op == 'd':
                output = self.dropout(output)
            elif op == 'n':
                output = self.layer_norm(output)

        return output


class Transformer_Encoder(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers,
                 relative_position, learnable_position, add_position,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 dropout=None, scaled=True, ff_size=-1,
                 mode=collections.defaultdict(bool),dvc=None,max_seq_len=-1,pe=None,
                 pe_ss=None,pe_se=None,pe_es=None,pe_ee=None,
                 k_proj=True,q_proj=True,v_proj=True,r_proj=True,
                 attn_ff=True,ff_activate='relu',lattice=False,
                 four_pos_shared=True,four_pos_fusion=None,four_pos_fusion_shared=True):
        '''

        :param input_size:
        :param hidden_size:
        :param num_layers:
        :param relative_position: bool
        :param learnable_position: bool
        :param add_position: bool, if False, concat
        :param layer_preprocess:
        :param layer_postprocess:
        '''
        super().__init__()
        self.four_pos_fusion_shared=four_pos_fusion_shared
        self.four_pos_shared = four_pos_shared
        self.four_pos_fusion = four_pos_fusion
        self.pe = pe
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        if self.four_pos_fusion_shared:
            self.four_pos_fusion_embedding = \
                Four_Pos_Fusion_Embedding(self.pe,self.four_pos_fusion,self.pe_ss,self.pe_se,self.pe_es,self.pe_ee,
                                          self.max_seq_len,self.hidden_size,self.mode)
        else:
            self.four_pos_fusion_embedding = None

        self.lattice = lattice
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.relative_position = relative_position
        if self.relative_position and self.lattice:
            assert four_pos_fusion is not None
        self.learnable_position = learnable_position
        self.add_position = add_position
        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        self.scaled = scaled
        self.k_proj=k_proj
        self.q_proj=q_proj
        self.v_proj=v_proj
        self.r_proj=r_proj
        self.attn_ff = attn_ff
        self.ff_activate = ff_activate

        if dvc is None:
            dvc = torch.device('cpu')
        self.dvc = dvc

        if self.relative_position and max_seq_len < 0:
            print_info('max_seq_len should be set if relative position encode')
            exit(1208)

        # if self.relative_position:
        #     print('现在还不支持相对编码！')
        #     exit(1208)

        # if not self.add_position:
        #     print('现在还不支持位置编码通过concat的方式加入')
        #     exit(1208)

        if dropout == None:
            dropout = collections.defaultdict(int)
        self.dropout = dropout

        if ff_size == -1:
            ff_size = hidden_size
        self.ff_size = ff_size

        for i in range(self.num_layers):
            setattr(self, 'layer_{}'.format(i),Transformer_Encoder_Layer(hidden_size, num_heads,
                                                    relative_position, learnable_position, add_position,
                                                    layer_preprocess_sequence, layer_postprocess_sequence,
                                                    dropout,scaled,ff_size,
                                                    mode=self.mode,
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
                                                    lattice=self.lattice,
                                                    four_pos_shared=self.four_pos_shared,
                                                    four_pos_fusion=self.four_pos_fusion,
                                                    four_pos_fusion_embedding=self.four_pos_fusion_embedding

                                                    ))

        self.layer_preprocess = Layer_Process(self.layer_preprocess_sequence,self.hidden_size)

    def forward(self, inp, seq_len,lex_num=0,pos_s=None,pos_e=None):
        output = inp
        if self.relative_position:
            if self.four_pos_fusion_shared and self.lattice:
                rel_pos_embedding = self.four_pos_fusion_embedding(pos_s,pos_e)
            else:
                rel_pos_embedding = None
        else:
            rel_pos_embedding = None
        for i in range(self.num_layers):
            now_layer = getattr(self,'layer_{}'.format(i))
            output = now_layer(output,seq_len,lex_num=lex_num,pos_s=pos_s,pos_e=pos_e,
                               rel_pos_embedding=rel_pos_embedding)

        output = self.layer_preprocess(output)

        return output





