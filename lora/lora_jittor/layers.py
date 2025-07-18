import math
from typing import Optional, List

import jittor as jt
import jittor.nn as nn

def _calculate_fan_in_and_fan_out(weight: jt.Var):
    """
    计算权重矩阵的 fan_in 和 fan_out，用于初始化权重。
    参数:
        weight: 2D 张量，形状为 (out_features, in_features)
    返回:
        fan_in (int): 输入特征数（in_features）
        fan_out (int): 输出特征数（out_features）
    异常:
        当张量维度小于 2 时抛出 ValueError
    """
    if weight.ndim < 2:
        raise ValueError("Fan in 和 fan out 需要至少二维张量")
    fan_in = weight.shape[1]
    fan_out = weight.shape[0]
    return fan_in, fan_out


class LoRALayer:
    """
    LoRA 基类，保存共有配置和状态。
    属性:
        r (int): LoRA 低秩秩 r
        lora_alpha (int): 缩放系数
        lora_dropout (函数或 nn.Module): 应用于 LoRA 的 dropout
        merge_weights (bool): 是否在评估模式下合并 LoRA 权重到原始权重
        merged (bool): 当前权重是否已合并
    """
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # 只有当 dropout 比例 > 0 时才使用 Dropout，否则使用恒等映射
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    """
    带 LoRA 的 Embedding 层，继承自 jittor.nn.Embedding。
    在原始 Embedding 输出上额外添加低秩适配参数 A 和 B:
        output = Embedding(x) + scaling * (Embedding_A(x) @ B)
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        # 初始化基础 Embedding
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        # 初始化 LoRA 参数和状态
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0., merge_weights=merge_weights)
        if r > 0:
            # A: (r × num_embeddings)，B: (embedding_dim × r)
            self.lora_A = jt.zeros((r, num_embeddings))
            self.lora_B = jt.zeros((embedding_dim, r))
            self.scaling = self.lora_alpha / self.r
            # 冻结原始权重，不参与梯度更新
            self.weight.stop_grad()
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """初始化原始 Embedding 和 LoRA 参数"""
        # 原始 weight 均匀初始化
        if hasattr(self, 'weight'):
            bound = 1 / math.sqrt(self.weight.shape[1])
            nn.init.uniform_(self.weight, -bound, bound)
        # LoRA A 初始化为 0，B 按高斯分布初始化
        if hasattr(self, 'lora_A'):
            jt.init.zero_(self.lora_A)
            jt.init.gauss_(self.lora_B, mean=0, std=1)

    def train(self, mode: bool = True):
        """
        切换训练/评估模式。
        如果 merge_weights=True，则在评估模式下将 LoRA 参数合并到 weight，
        在训练模式下将合并权重撤销。
        """
        super().train(mode)
        if mode:
            # 切换到训练模式，若已合并则撤销合并
            if self.merge_weights and self.merged and self.r > 0:
                delta = jt.transpose(self.lora_B @ self.lora_A, 0, 1) * self.scaling
                self.weight.data -= delta
                self.merged = False
        else:
            # 切换到评估模式，若未合并则合并
            if self.merge_weights and not self.merged and self.r > 0:
                delta = jt.transpose(self.lora_B @ self.lora_A, 0, 1) * self.scaling
                self.weight.data += delta
                self.merged = True

    def execute(self, x: jt.Var):
        """
        前向计算:
        如果 r>0 且未合并，则计算基础 Embedding 输出加 LoRA 增量；
        否则直接返回基础 Embedding 输出。
        """
        if self.r > 0 and not self.merged:
            # 基础 Embedding 输出
            result = nn.embedding(x, self.weight, self.padding_idx)
            # 先用 A 投影，再用 B 恢复
            after_A = nn.embedding(x, jt.transpose(self.lora_A, 0, 1), self.padding_idx)
            lora_part = jt.transpose(after_A @ self.lora_B, 0, 1) * self.scaling
            return result + lora_part
        else:
            return nn.embedding(x, self.weight, self.padding_idx)


class Linear(nn.Linear, LoRALayer):
    """
    带 LoRA 的 Linear 层，继承自 jittor.nn.Linear。
    在执行基础线性变换后，加上 LoRA 部分:
        result = Linear(x) + scaling * dropout(x) @ A^T @ B^T
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        # 初始化基础 Linear
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out

        if r > 0:
            # A: (r × in_features), B: (out_features × r)
            self.lora_A = jt.zeros((r, in_features))
            self.lora_B = jt.zeros((out_features, r))
            self.scaling = self.lora_alpha / self.r
            # 冻结原始权重
            self.weight.stop_grad()

        # 参数初始化
        self.reset_parameters()
        # 如果权重存储格式为 (in, out)，需要转置
        if fan_in_fan_out:
            self.weight.data = jt.transpose(self.weight.data, 0, 1)

    def reset_parameters(self):
        """初始化 weight, bias, LoRA A 和 B"""
        if hasattr(self, 'weight'):
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.weight, -bound, bound)
        if hasattr(self, 'bias') and self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if hasattr(self, 'lora_A'):
            jt.init.relu_invariant_gauss_(self.lora_A, mode="fan_in")
            jt.init.zero_(self.lora_B)

    def train(self, mode: bool = True):
        """
        切换训练/评估模式，训练时撤销合并，评估时合并权重。
        """
        super().train(mode)
        def T(w): return jt.transpose(w, 0, 1) if self.fan_in_fan_out else w

        if mode:
            # 训练模式，若已合并则撤销
            if self.merge_weights and self.merged and self.r > 0:
                delta = T(self.lora_B @ self.lora_A) * self.scaling
                self.weight.data -= delta
                self.merged = False
        else:
            # 评估模式，若未合并则合并
            if self.merge_weights and not self.merged and self.r > 0:
                delta = T(self.lora_B @ self.lora_A) * self.scaling
                self.weight.data += delta
                self.merged = True

    def execute(self, x: jt.Var):
        """
        前向计算:
        如果 r>0 且未合并，先做基础 Linear，再加 LoRA 增量；
        否则直接调用父类 execute。
        """
        if self.r > 0 and not self.merged:
            orig_shape = x.shape  # [batch, seq_len, in_features]
            x_2d = x.reshape([-1, orig_shape[-1]])  # 展平
            # 计算 LoRA 增量
            A_T = self.lora_A.transpose(1, 0)  # [in_features, r]
            B_T = self.lora_B.transpose(1, 0)  # [r, out_features]
            part = self.lora_dropout(x_2d)
            part = jt.matmul(part, A_T)
            part = jt.matmul(part, B_T)
            part = part.reshape(*orig_shape[:-1], self.weight.shape[0]) * self.scaling
            # 基础 Linear 输出
            result = super(Linear, self).execute(x)
            return result + part
        else:
            return super(Linear, self).execute(x)


class MergedLinear(nn.Linear, LoRALayer):
    """
    支持分组 LoRA 的 Linear 层，可对部分输出通道启用 LoRA。
    训练/评估模式下支持合并与拆分。
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, "enable_lora 长度必须可整除 out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out

        if r > 0 and any(enable_lora):
            # 按组初始化 A 和 B
            self.lora_A = jt.zeros((r * sum(enable_lora), in_features))
            self.lora_B = jt.zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            self.scaling = self.lora_alpha / self.r
            self.weight.stop_grad()  # 冻结原始
            # 构造布尔索引，标识启用 LoRA 的通道
            ind = jt.zeros((out_features,), dtype=jt.bool).view(len(enable_lora), -1)
            ind[enable_lora, :] = True
            self.lora_ind = ind.view(-1)

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = jt.transpose(self.weight.data, 0, 1)

    def reset_parameters(self):
        """初始化权重和 LoRA 参数"""
        if hasattr(self, 'weight'):
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.weight, -bound, bound)
        if hasattr(self, 'bias') and self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if hasattr(self, 'lora_A'):
            jt.init.relu_invariant_gauss_(self.lora_A, mode="fan_in")
            jt.init.zero_(self.lora_B)

    def zero_pad(self, x: jt.Var):
        """
        将小的 LoRA 增量张量按布尔索引填充回完整输出维度。
        """
        result = jt.zeros((len(self.lora_ind), *x.shape[1:]), dtype=x.dtype)
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        """
        计算并返回合并后的增量权重 delta_w = B @ A，
        并按需转置和填充到完整形状。
        """
        def T(w): return jt.transpose(w, 0, 1) if self.fan_in_fan_out else w
        # 使用一维卷积计算 B @ A
        delta = nn.conv1d(
            self.lora_A.unsqueeze(0),
            self.lora_B.unsqueeze(-1),
            groups=sum(self.enable_lora)
        ).squeeze(0)
        delta_w = T(self.zero_pad(delta))
        return delta_w

    def train(self, mode: bool = True):
        """
        切换训练/评估模式，支持合并与拆分增量权重。
        """
        super().train(mode)
        def T(w): return jt.transpose(w, 0, 1) if self.fan_in_fan_out else w
        if mode:
            # 训练模式，若已合并则撤销
            if self.merge_weights and self.merged and self.r > 0 and any(self.enable_lora):
                self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            # 评估模式，若未合并则合并
            if self.merge_weights and not self.merged and self.r > 0 and any(self.enable_lora):
                self.weight.data += self.merge_AB() * self.scaling
                self.merged = True

    def execute(self, x: jt.Var):
        """
        前向计算: 如果已合并增量，则直接调用父类；
        否则先计算基础输出，再加 LoRA 增量。
        """
        def T(w): return jt.transpose(w, 0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return nn.Linear.__call__(self, x)
        # 基础输出
        result = nn.Linear.__call__(self, x)
        if self.r > 0 and any(self.enable_lora):
            part = self.lora_dropout(x) @ T(self.merge_AB().transpose(0,1)) * self.scaling
            return result + part
        return result


class ConvLoRA(nn.Module, LoRALayer):
    """
    带 LoRA 的卷积层包装，可包装任意 nn.ConvXd 模块。
    支持在评估时合并参数。
    """
    def __init__(
        self,
        conv_module,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        super().__init__()
        # 原始卷积层
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.merged = False
        assert isinstance(kernel_size, int), "kernel_size 必须为整数"
        if r > 0:
            # A: (r*kernel_size) × (in_channels*kernel_size)
            self.lora_A = jt.zeros((r * kernel_size, in_channels * kernel_size))
            # B: (out_channels/groups*kernel_size) × (r*kernel_size)
            self.lora_B = jt.zeros((out_channels // self.conv.groups * kernel_size, r * kernel_size))
            self.scaling = self.lora_alpha / self.r
            self.conv.weight.stop_grad()
        self.reset_parameters()

    def reset_parameters(self):
        """初始化原始卷积和 LoRA 参数"""
        if hasattr(self.conv, 'reset_parameters'):
            self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            jt.init.relu_invariant_gauss_(self.lora_A, mode="fan_in")
            jt.init.zero_(self.lora_B)

    def train(self, mode: bool = True):
        """
        切换训练/评估模式，支持增量权重的合并与拆分。
        """
        super().train(mode)
        if mode:
            # 训练模式，若已合并则撤销
            if self.merge_weights and self.merged and self.r > 0:
                delta = (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.conv.weight.data -= delta
                self.merged = False
        else:
            # 评估模式，若未合并则合并
            if self.merge_weights and not self.merged and self.r > 0:
                delta = (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.conv.weight.data += delta
                self.merged = True

    def execute(self, x: jt.Var):
        """
        前向计算: 如果 r>0 且未合并，则使用合并权重执行卷积；
        否则直接调用原始卷积。
        """
        if self.r > 0 and not self.merged:
            delta_w = (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
            return self.conv.execute_with_weight(x, self.conv.weight + delta_w, self.conv.bias)
        return self.conv(x)


class Conv2d(ConvLoRA):
    """包装 nn.Conv2d 的 ConvLoRA"""
    def __init__(self, *args, **kwargs):
        super().__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(ConvLoRA):
    """包装 nn.Conv1d 的 ConvLoRA"""
    def __init__(self, *args, **kwargs):
        super().__init__(nn.Conv1d, *args, **kwargs)


class Conv3d(ConvLoRA):
    """包装 nn.Conv3d 的 ConvLoRA"""
    def __init__(self, *args, **kwargs):
        super().__init__(nn.Conv3d, *args, **kwargs)