from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import init

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
        # dropout值
        self.dropout = dropout

        self.bos_token_id = bos_token_id

        self.eos_token_id = eos_token_id

        self.hidden_act = hidden_act
        # 维度D
        self.hidden_size = hidden_size
        # 进行FFN时的升维后的中间维度
        self.intermediate_size = intermediate_size
        # 最大的上下文长度
        self.max_position_embeddings = max_position_embeddings
        # Q的头数 
        self.num_attention_heads = num_attention_heads
        # KV的头数
        self.num_key_value_heads = num_key_value_heads
        # Transformer的(GQA+FFN)层数
        self.num_hidden_layers = num_hidden_layers
        # 词典的大小
        self.vocab_size = vocab_size
        # rms的分母中防止除数为0的eps
        self.rms_norm_eps = rms_norm_eps
        # 公式中theta = b^(-2i / d)中的b
        self.rope_theta = rope_theta
        # 是否使用yarn进行推理
        self.inference_rope_scaling = inference_rope_scaling
        # 是否使用flash attention
        self.flash_attention = flash_attention

        self.use_moe = use_moe
        # 每个token几个专家
        self.num_experts_per_tok = num_experts_per_tok
        # 一共有几个路由专家
        self.n_routed_experts = n_routed_experts
        # 一共有几个共享专家
        self.n_shared_experts = n_shared_experts
        # 是否使用序列级别的辅助损失
        self.seq_aux = seq_aux
        # 是否对topk的概率进行归一化
        self.norm_topk_prob = norm_topk_prob
        # 辅助损失的权重
        self.aux_loss_alpha = aux_loss_alpha
        # scoring_func的选择 默认softmax
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
from typing import List, Optional, Tuple, Union
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
            else args.num_key_value_heads
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
                # Bd -> BHLd
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

from transformers.activations import ACT2FN
class FeedForward(nn.Module):
    def __init__(self,  config:MymindConfig):
        super().__init__()
        if config.intermediate_size is None:
            # 计算合适的中间层维度 64的整数倍
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear( config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x))))

class MoEGate(nn.Module):
    def __init__(self, config: MymindConfig):
        super().__init__()
        self.config = config
        # 每个token几个专家
        self.top_k = config.num_experts_per_tok
        # 一共有几个专家
        self.n_routed_experts = config.n_routed_experts
        # softmax
        self.scoring_func = config.scoring_func
        # loss的权重
        self.alpha = config.aux_loss_alpha
        # 是否在序列级别计算辅助损失
        self.seq_aux = config.seq_aux
        # 是否对topk的概率进行归一化
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        # nn.Parameter注册为参数，并且初始化权重矩阵，大小为 (专家数量n, D)
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        B, L, D = hidden_states.shape
        # (token数量, D) -> (token数量, n_routed_experts)
        hidden_states = hidden_states.view(-1, D)
        # F.linear(input, weight, bias) 相当于 hidden_states @ self.weight.T
        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == "softmax":
            # scores.shape = (token数量, 专家数量)
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )
        
        # 从指定的维度中找到输入的最大的k个值以及索引位置
        # topk_weight.shape = topk_idx.shape = (token数量, topk个专家的数量)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        # 对每个token的选择的topk个专家的概率进行归一化，使得它们的和为1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores # shape = (token数量, 专家数量)
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(B, -1) # (B, L * topk)
            if self.seq_aux:
                # shape = (B, L, n_routed_experts)
                scores_for_seq_aux = scores_for_aux.view(B, L, -1)
                # ce.shape = (B, n_routed_experts) 用来记录每个句子的每个专家被选中的次数
                ce = torch.zeros(
                    B, self.n_routed_experts, device=hidden_states.device
                )
                # 把src中的按照topk_idx_for_aux_loss中的索引位置，加到ce对应的位置上，表示该专家被选中了一次
                ce.scatter_add_(
                    dim=1,
                    index=topk_idx_for_aux_loss,
                    # 按照index位置决定每次加多少
                    src=torch.ones(B, L * aux_topk, device=hidden_states.device),
                ).div_(L * aux_topk / self.n_routed_experts) #除数是一个句子中(L)每个专家的理想平均负载 
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                # mask_ce.shape = (token数量, n_routed_experts) 用来记录每个token的每个专家被选中的情况
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts # 指定 one-hot 长度
                )
                # 计算每个专家的平均负载情况
                ce = mask_ce.float().mean(0)
                # 计算每个专家被选中的概率
                Pi = scores_for_aux.mean(0)
                # 计算每个专家的负载
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            # 创建一个与 scores 具有相同设备和数据类型的标量张量 aux_loss，并将其初始化为 0
            # new_zeros(1) 创建一个包含单个元素的张量，squeeze() 将其从形状 (1,) 转换为标量
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss

class MoEFeedForward(nn.Module):
    def __init__(self, config: MymindConfig):
        super().__init__()
        self.config = config
        # 专家层
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )
        # 门控层
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape

        # 使用门控机制选择专家
        # topk_idx和topk_weight的形状都是(B*L, K)
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # x.shape = (B*L, D)
        x = x.view(-1, x.shape[-1])

        flat_topk_idx = topk_idx.view(-1) # （B*L*K）
        if self.training:
            # 按照定义的num_experts_per_tok重复输入token
            # torch.repeat_interleave(input, repeats, dim=None) repeats指定沿着dim维度重复的次数
            x = x.repeat_interleave(repeats=self.config.num_experts_per_tok, dim=0) # （B*L*K,D）
            # y要存放最后的专家输出结果，形状和x一样
            y = torch.empty_like(x, dtype=x.dtype)
            # 遍历所有专家
            for i, expert in enumerate(self.experts):
                # 将当前专家的输入数据传入专家网络，得到输出结果
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    # 如果当前专家没有被任何token选择，则将y中对应位置的值设置为0，并加上一个与专家参数相关的零张量，使得专家参数加入计算图
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(
                        p.sum() for p in expert.parameters()
                    )
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1) # 在进行sum之前维度：(B*L, K, D)，经过sum之后维度（B*L, D）
            y = y.view(*orig_shape) # (B, L, D)
        # 如果是推理阶段
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 使用cache，创建一个和x形状相同的零张量，用于存储最终的专家的输出结果
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序，最后是[0,0,0,1,1,2,2,2,...]这样的顺序
        idxs = flat_expert_indices.argsort() # 得到的是从小到大的位置索引（后续 //k 可以得到token的索引）
        # 计算每个专家的token数量，并进行累加，得到每个专家的结束位置索引
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个token对应的专家索引
        token_idxs = idxs // self.config.num_experts_per_tok
        # 对每个打包好的包进行处理
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前包的起始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 取出当前包对应的专家
            expert = self.experts[i]
            # 取出token对应的原始id
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 取出token对应的数据
            expert_tokens = x[exp_token_idx] # (N个token,D)
            # 计算专家输出，一次性处理当前包的所有token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 加权
            expert_out.mul_(flat_expert_weights[idxs[st00art_idx:end_idx]]) #(N,D)
            # 将结果散点加到缓存中对应位置
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache

class MymindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MymindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attention = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = (
            FeedForward(config)
            if not config.use_moe
            else MoEFeedForward(config)  
        )

    def forward(
        self,
        hidden_states,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        res = hidden_states

        hidden_states, present_key_value = self.self_attention(
            self.input_layernorm(hidden_states),  # pre-norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )

        hidden_states = res + hidden_states

        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value

class MymindModel(nn.Module):
    def __init__(self, config: MymindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [MymindBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        # input_ids: [B,L]
        batch_size, seq_length = input_ids.shape
        # 如果传进来的 past_key_values 不是你这份代码期望的“List[Tuple[K,V]]”格式，
        # 而是某种带 .layers 属性的 Cache 对象，就先把它丢掉，重新按自己的格式处理
        if hasattr(past_key_values, "layers"):
            past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算start_pos：如果存在past，则start_pos为已有past序列长度
        start_pos = (
            # KV 的 shape B L_KV H D
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # Embedding + dropout
        hidden_states = self.dropout(
            self.embed_tokens(input_ids)
        )  # [B, L, D]

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )

        presents = []
        # 让 hidden_states 依次通过每一层 Transformer Block，
        # 同时给每一层传入它自己的 KV Cache，并收集新的 KV Cache
        for (layer, past_key_value) in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
        # B L D
        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            [
                layer.mlp.aux_loss
                for layer in self.layers
                if isinstance(
                    layer.mlp, MoEFeedForward
                )
            ],
            hidden_states.new_zeros(1).squeeze(),
        )

        return hidden_states, presents, aux_loss

class MymindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MymindConfig

    def __init__(self, config: MymindConfig):
        super().__init__(config)
        self.model = MymindModel(config)
        # 映射到词表的维度得到每个token的预测分数
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Weight Tying（权重共享）
        # 输入 Embedding 的权重 = 输出 lm_head 的权重
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )
        #只对指定位置的 hidden states 计算词表 logits，从而减少不必要的计算
        slice_indices = (
            #logits_to_keep 是这个 forward 的参数，用来指定：对哪些 token 位置计算词表预测分数（logits）
            slice(-logits_to_keep, None) # slice(start, stop, step)分别表示起始位置、结束位置（不包含）、步长。省略步长时默认为 1，结束位置为 None 时表示一直取到末尾
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        # 取指定的 token 位置
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # 输入:   A   B   C   D
            # 预测:   B   C   D
            # 错开预测和标签
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                x.view(-1, x.size(-1)),# (B*(L-1),vocab_size)
                y.view(-1), # (B*(L-1))
                ignore_index=-100,  # label为100的标签不参与计算！
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output

