from transformers import PretrainedConfig


class MymindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

        # Q以及KV的头数 
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func
        # yarn的参数
        self.rope_scaling = (
            {
                "beta_fast": 32, # hiddendim中的i如果在训练的时候转的圈数大于beta_fast则不用线性缩放
                "beta_slow": 1, # hiddendim中的i如果在训练的时候转的圈数小于beta_slow则完全线性缩放theta_i / s(factor)
                "factor": 16, # s = L' / L
                "original_max_position_embeddings": 2048, # 原始训练时的最长token数量
                "attention_factor": 1.0, # attention计算的时候的参数t
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, hidden_dim:int, eps:float=1e-6):
        super().__init__()
        # # x.shape = (B, L, D)
        # B: batch_size，一批中有多少条 sequence
        # L: seq_len，每条 sequence 中有多少个 token
        # D: hidden_dim，每个 token 的 hidden state 有多少维
        # hidden_dim表示token维度
        self.dim = hidden_dim
        # 防止除以0
        self.eps = eps
        # # 每个 hidden dimension 都有一个独立的可学习缩放参数
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        # 平方均值（Mean Square）
        mean_squared = x.pow(2).mean(
            dim = -1, # 在BLD的最后一个D维度上进行计算
            keepdim = True # 计算完之后保留最后一个维度BLD->BL1
        )
        rsqrt = torch.rsqrt(mean_squared + self.eps)
        x_norm = x * rsqrt
        output = self.weight * x_norm
        return output

# 先写yarn方法
import math
from typing import Optional, Tuple
from torch.nn import functional as F
def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end > orig_max:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # (x,y) -> (-y,x)
    def rotate_half(x):
        return torch.cat((-x[..., x.shape(-1)//2:], x[...,:x.shape[-1]//2]),dim=-1)
    # (BHD) -> (BLHD)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # (xcos,ycos) + (-ysin,xsin) = (xcos-ysin,xsin+ycos)
    q_embed = q*cos + rotate_half(q)*sin
    k_embed = k*cos + rotate_half(k)*sin
    return q_embed, k_embed

def repeat_kv(x:torch.Tensor, n_rep:int)->torch.Tensor:
    B, L, H, d = x.shape
    if n_rep == 1:
        return x
    return(
        x[:,:,:,None,:].expand(B,L,H,n_rep,d).reshape(B,L,H*n_rep,d)
    )

class Attention(nn.Module):
    def __init__(self, args:MymindConfig):
        super().__init__()
        # 有GQA用GQA 没有 就用MHA
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key
        )
        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // self.n_local_heads

        self.q_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 出投影层。把多头 Attention 拼接后的结果再做一次线性变换
        self.o_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        # 对 Attention 权重 softmax(QKᵀ) 做 dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 对 Attention 最终输出 做 dropout，再送到残差连接
        self.resid_dropout = nn.Dropout(args.dropout)
        # 只是保存 dropout 概率，例如 0.1，主要给 Flash Attention 使用
        self.dropout = args.dropout
        # 判断是否可以使用 PyTorch 的 Flash/SDPA Attention 加速实现。
        # 要求 PyTorch 有 scaled_dot_product_attention，并且配置里 flash_attention=True
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

    def forward(
        self, x:torch.Tensor,position_embeddings:Tuple[torch.Tensor,torch.Tensor],
        past_key_value:Optional[Tuple[torch.Tensor,torch.Tensor]] = None,
        use_cache=False,
        attention_mask:Optional[torch.Tensor] = None
    ):
        # x.shape = (B,L,D)
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        cos, sin = position_embeddings 
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # kv_cache实现
        if past_key_value is not None:
            # 在句子长度S上进行拼接！
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 注意力分数计算QK^T需要保持维度一致,同时将QKV转换维度为(BHLD)
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        if (
            self.flash
            and (seq_len > 1) # 不只是单 token 推理
            and (past_key_value is None) # 没使用 KV Cache
            and (attention_mask is None or torch.all(attention_mask == 1)) # 没有 padding 之类的特殊 mask
        ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # scores.shape = (BHLL)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # -seq_len: = 只对当前新增 token 对应的最后几列 K 做 causal mask
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),# 将(L,L)矩阵填充"-inf"
                diagonal=1,# 对角线以上不包括对角线保留原来的值,其余为0
            )
            # 把 padding 位置屏蔽掉，让模型不要关注无效 token
            if attention_mask is not None:
                # BS -> BHLS
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            # 沿最后一个维度，也就是“对每个 Q 的所有 K 分数做 softmax”
            # 用 float32 安全地算 softmax，再恢复成 Q 原来的数据类型
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
