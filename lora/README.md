# LoRA 微调对比实验

本文基于 PyTorch 与 Jittor 两种深度学习框架，分别实现了 DistilBERT + LoRA 低秩适配层的微调，并在 SST-2 数据集上对比二者在训练速度、资源占用以及模型收敛效果上的差异，结合代码实现细节给出深入分析与优化建议。

## 实验设置

- **数据集**：HuggingFace SST-2 已预处理数据（input_ids、attention_mask、label）  
- **模型结构**：DistilBERT Base + LoRA（仅微调 Q/V 矩阵的低秩增量和最终分类头）  
- **超参数**：  
  - batch_size = 16  
  - gradient_accumulation_steps = 2  
  - learning_rate = 2e-4  
  - num_train_epochs = 5  
- **精度模式**：  
  - PyTorch：混合精度（`torch.cuda.amp`）  
  - Jittor：全 FP32  

---

## 一、性能与时长对比

| 框架    | 第 1 Epoch (s) | 稳态 Epoch (s) | 关键配置                                      |
|-------|---------------|--------------|-------------------------------------------|
| PyTorch | ~439          | ~427         | `torch.cuda.amp` + `num_workers=4` + `pin_memory=True` |
| Jittor  | ~455          | ~415         | 全 FP32 + 默认单进程 I/O                       |

**详细分析**  
- PyTorch AMP 在前向/反向传播中将部分激活与梯度存储为 FP16，带来显著带宽节省；  
- 但每步需执行 `scaler.scale()`、`unscale_()`、`scaler.update()`，并且跨线程调用 `nvidia-smi` 会产生轻微阻塞；  
- Jittor 默认全 FP32，无混合精度开销，底层算子融合（Fusion）和调度（Scheduling）让 GPU 队列更饱满；  
- 随着迭代进入“稳态”，Jittor 调度开销进一步摊薄，Epoch 时长下降幅度 (~9%) 高于 PyTorch (~3%)。

---

## 二、显存与 GPU 利用率

| 框架    | 平均显存 (GB) | 平均利用率 (%) | 原因说明                                         |
|-------|-------------|-------------|----------------------------------------------|
| PyTorch | ~1.8         | ~70–72      | 混合精度降内存 ~50%；数据加载线程与 GPU 计算交替，导致利用率波动 |
| Jittor  | ~3.8         | ~91–95      | 全 FP32 模式；同步 I/O 与 Compute，无多线程切换开销               |

**细节探究**  
1. PyTorch：  
   - `torch.cuda.amp.autocast()` 自动使用半精度计算，减少激活内存；  
   - DataLoader 多线程预取在加载较慢时会让 GPU 空转；  
2. Jittor：  
   - `jt.flags.use_cuda = 1` 开启全流程 CUDA 加速；  
   - 单进程前向/反向与数据 I/O 串行，但支持算子融合，减少内存拷贝，显存占用翻倍同时利用率提升 ~25%。

---

## 三、训练效果对比

| 框架    | 最终 Accuracy | 最终 Loss   |
|-------|-------------|-----------|
| PyTorch | ~0.95–0.97   | ~0.12–0.15 |
| Jittor  | ~0.96–0.98   | ~0.05–0.10 |

**原因探讨**  
- 两者均只微调 LoRA 增量权重和分类头（`mark_only_lora_as_trainable` / `get_peft_model([...])`）；  
- PyTorch AMP 在极端小梯度或梯度累积边缘可能出现数值抖动；  
- Jittor 全 FP32 下，自定义 `Linear` 与 `MultiHeadAttention` 在 softmax/dropout、LayerNorm 顺序上有微调，带来更平滑的收敛。

---

## 四、总结与优化建议

1. **精度与显存折中**  
   - 显存充足时，可在 PyTorch 中关闭 AMP（`autocast`）或强制全 FP32；  
   - 显存受限时，继续使用 PyTorch AMP，或在 Jittor 中引入混合精度（`jt.cuda_half()`）。  
2. **数据加载优化**  
   - PyTorch DataLoader：调节 `num_workers`、`pin_memory`，或使用 `IterableDataset` 降低拷贝延迟；  
   - Jittor：尝试多线程管道或预分配缓存队列。  
3. **算子调度优化**  
   - 借鉴 Jittor 算子融合：PyTorch 可开启 `torch.backends.cudnn.benchmark=True`、使用 `torch.jit.script`；  
   - PyTorch 可微调 LoRA dropout、梯度裁剪、LR 调度等以稳定 AMP 训练。  
4. **监控与调试**  
   - 避免训练循环频繁调用 `nvidia-smi`，可用 PyTorch Profiler 或 `torch.cuda.max_memory_allocated()`；  
   - Jittor 中启用 `jt.flags.verbose=1` 查看底层调度日志，定位瓶颈。

---

## 五、Windows 下的多进程 DataLoader 与自动混合精度问题

### 5.1 多进程 DataLoader 共享内存失败

**错误信息**  
```text
OSError: [WinError 1455] 页面文件太小，无法完成操作
...
File "...\\shared_memory.py", line 131, in __init__
  h_map = _winapi.CreateFileMapping(...)
OSError: [WinError 1455] 页面文件太小，无法完成操作: 'wnsm_xxx'
```

**原因**  
- Windows 共享内存机制受页面文件大小限制，Jittor 的 `DataLoader(num_workers>0)` 使用 `multiprocessing.shared_memory`，在 Windows 下兼容性较差，容易触发分配失败。

**解决建议**  
1. 临时：将 `num_workers=0`，使用单进程加载。  
2. 持久：增大系统虚拟内存（页面文件）或切换到 Linux 环境运行多进程加载。

---

### 5.2 自动混合精度运行时/编译错误

**错误信息**  
```text
RuntimeError: Wrong inputs arguments, Please refer to examples(help(jt.sync)).
Types of your inputs are:
  self = module, args = (list, )
...
Error happend during ring compilation:
  [Error] ... broadcast_to__Tx_float32_DIM_3 ...
Reason: Check failed: a->dtype() == b->dtype() Something wrong...
```

**原因**  
- Jittor 在自动混合精度流程中，会插入 `jt.sync` 调用，参数类型与接口声明不符；  
- 算子融合阶段，不同数据类型（float16/float32）的输入或缓冲区混用，触发 dtype 检查失败。

**解决建议**  
1. 临时：关闭自动混合精度  
   ```python
   with jt.flag_scope(auto_mixed_precision_level=0):
       # 关闭 AM ...
   ```  
2. 手动：将模型及输入全部转换为 `float16`，并确保所有计算一致：  
   ```python
   model.float16()
   ```  
3. 长期：升级到最新 Jittor 版本，或关注官方修复 dtype mismatch 问题。
