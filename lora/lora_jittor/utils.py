import jittor as jt                # Jittor 主框架，用于张量和自动微分
import jittor.nn as nn             # Jittor 神经网络模块
from typing import Dict           # 用于类型注解

from .layers import LoRALayer      # 导入 LoRA 层基类，用于类型检查

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """
    冻结模型中所有非 LoRA 参数，仅保留 LoRA 模块（及可选偏置）可训练。

    参数:
        model (nn.Module): 待处理的 Jittor 模型。
        bias (str): 偏置项训练策略，取值:
            'none'      - 禁用所有偏置项梯度（默认）。
            'all'       - 所有偏置项均可训练。
            'lora_only' - 仅 LoRA 层中的 bias 可训练。
    返回:
        None (就地修改 model 参数的 requires_grad 状态)。
    """

    # 1. 遍历模型所有参数，冻结非 LoRA 参数
    #    named_parameters 返回 (name, 参数 Tensor) 对序列
    for name, param in model.named_parameters():
        # 只要名字中不包含 'lora_'，就停止其梯度计算
        if 'lora_' not in name:
            param.stop_grad()

    # 2. 根据 bias 参数决定偏置项是否恢复可训练
    if bias == 'none':
        # 不解冻任何 bias，直接返回
        return

    elif bias == 'all':
        # 将所有名字中包含 'bias' 的参数恢复训练
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.start_grad()

    elif bias == 'lora_only':
        # 仅解冻那些属于 LoRALayer 的偏置
        for module in model.modules():
            if isinstance(module, LoRALayer) and hasattr(module, 'bias') and module.bias is not None:
                module.bias.start_grad()

    else:
        # 不支持的 bias 模式，抛出异常
        raise NotImplementedError(f"Unsupported bias mode: {bias}")


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, jt.Var]:
    """
    从模型 state_dict 中筛选并返回 LoRA 相关参数（可选含偏置）。

    参数:
        model (nn.Module): 待提取参数的模型。
        bias (str): 偏置项筛选策略，取值:
            'none'      - 仅返回 LoRA 权重（'lora_' 前缀）。
            'all'       - 返回所有 LoRA 权重和所有模型偏置。
            'lora_only' - 返回 LoRA 权重及 LoRA 模块自带 bias。
    返回:
        Dict[str, jt.Var]: 筛选后的参数字典，键为 state_dict 中的名称，值为对应张量。
    """

    # 获取完整 state_dict，包含所有参数
    full_state = model.state_dict()

    if bias == 'none':
        # 只保留名字中含 'lora_' 的参数
        return {k: full_state[k] for k in full_state if 'lora_' in k}

    elif bias == 'all':
        # 保留 LoRA 权重及所有带 bias 的参数
        return {
            k: full_state[k]
            for k in full_state
            if 'lora_' in k or 'bias' in k
        }

    elif bias == 'lora_only':
        # 保留 LoRA 权重，并为每个 LoRA 权重对应的模块 bias 一起保留
        collected: Dict[str, jt.Var] = {}
        for key in full_state:
            if 'lora_' in key:
                # 添加 LoRA 权重
                collected[key] = full_state[key]
                # 构造对应偏置名称：将 'lora_' 前缀前的部分 + 'bias'
                prefix = key.split('lora_')[0]
                bias_name = prefix + 'bias'
                # 如果该偏置存在于 state_dict 中，则一并添加
                if bias_name in full_state:
                    collected[bias_name] = full_state[bias_name]
        return collected

    else:
        # 不支持的 bias 策略
        raise NotImplementedError(f"Unsupported bias mode: {bias}")