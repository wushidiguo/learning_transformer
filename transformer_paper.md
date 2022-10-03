# Transformer系列经典论文笔记

## [Attention is All You Need](papers/attention%20is%20all%20you%20need.pdf)

\#**transformer**    \#**self-attention**    \#**machine translation**

- 提出了完全基于注意力机制的Transformer模型，相较RNN具有更高的computational efficiency和scalability，可以训练更大的模型

- 使用encoder-decoder架构（基于机器翻译任务）

- 使用multi-head attention，自注意力的输入和输出维度保持不变

- decoder使用masked attention

- 使用positional encoding
  
  ### 模型架构
  
  ![transformer.png](images/transformer.png)
  
  ### pytorch实现
  
  [github链接](https://github.com/wushidiguo/transformer)
  
  ---

## [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](papers/BERT.pdf)

\#**BERT**    \#**mask**    #**self-supervised**    #**NLP**

- 使用transformer的encoder学习language representations

- 通过两个无监督学习任务进行预训练，Masked LM和Next Sentence Prediction (NSP)

- 所有输入都前置特殊字符[cls]，其对应的输出代表整个序列，用于分类任务

- 对下游任务，以预训练模型参数作为初始化，在有标签数据上对全部参数进行fine-tuning

- BERT<sub>BASE</sub> (L=12, H=768, A=12, Total Parameters=110M) and BERT<sub>LARGE</sub> (L=24, H=1024, A=16, Total Parameters=340M)

- 使用GeLU激活函数
  
  ### 模型架构
  
  ![bert_training.png](images/bert_training.png)
  
  ### pytorch实现
  
  [github链接](https://github.com/wushidiguo/BERT-pytorch)
  
  ---

## [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](papers/ViT.pdf)

\#**ViT**    \#**CV**

- 使用标准的transformer encoder，ViT-B和ViT-L分别对应BERT<sub>BASE</sub>和BERT<sub>LARGE</sub>
  
  | Model     | Layers | Hidden size D | Heads | Params |
  | --------- |:------:|:-------------:| ----- |:------:|
  | ViT-Base  | 12     | 768           | 12    | 86M    |
  | ViT-Large | 24     | 1024          | 16    | 307M   |
  | ViT-Huge  | 32     | 1280          | 16    | 632M   |
  
  ViT-L/16指patch size为16 × 16的ViT-Large模型

- 将图像分割成固定大小的patches，每个patch拉直后经过线性embedding并加上位置编码，作为transformer的输入

- 采用learnable 1D位置编码，patch之间的空间关系通过学习得到

- fine-tuning时，采用0值初始化的D × K的前馈层代替预训练的分类头；在比预训练阶段更高分辨率的图像上进行微调效果更好，保持patch尺寸不变，增大序列长度，预训练得到的位置编码需要进行2D插值以适应此变化

- 相比于CNNs，transformer缺少相应的归纳偏置（translation equivariance and locality），在训练数据集较小的时候，表现不及CNNs；当在足够大数据集（14M-300M images）上进行预训练，再迁移到小数据集上时，表现优异

- 序列长度与patch size的平方成反比，patch size越小，计算越昂贵
  
  ### 模型架构
  
  ![vit.png](images/vit.png)
  
  ### pytorch实现
  
  [github链接](https://github.com/wushidiguo/vit-pytorch)
  
  ---

## [Masked Autoencoders Are Scalable Vision Learners](papers/MAE.pdf)

\#**MAE**    \#**CV**    \#**autoencoder**    \#**self-supervised**

- 使用非对称的encoder-decoder架构进行预训练，只有可见的patches会通过encoder，得到的latent representation和mask tokens一起作为decoder的输入，重建原图像；训练完成后，只有encoder用于下游任务的fine-tuning或linear-probing

- 相较于自然语言，图像有严重的信息冗余，使用较高的masking ratio（e.g.,75%)，可以迫使模型从整体理解图像内容，而且可以节省3×以上的训练时间和内存占用

- 损失函数为重建图像与原图像的均方误差（MSE），只计算masked patches的损失

- 使用轻量级的decoder，可以是单个block的transfomer

- mask token全局共享，通过学习得到

- 与对比学习不同的是，不使用数据增强，不会显著影响MAE的准确率。随机的masking在每次迭代中产生不同样本，降低了对数据增强的依赖

- 相较于ViT-L有监督学习，可以在较小的数据集上（ImageNet-1K）开展预训练，取得更好的泛化效果

### 模型架构

![mae.png](D:\notebook\deep-learning-notebook\images\mae.png)

### pytorch实现

- [github链接](https://github.com/wushidiguo/vit-pytorch)

- 实现随机mask的部分代码：
  
  ```python
  rand_indices = torch.rand(batch, num_patches).argsort(dim = -1)
  masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
  # get the unmasked tokens to be encoded
  batch_range = torch.arange(batch)[:, None]
  tokens = tokens[batch_range, unmasked_indices]
  ```

---

## [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](papers/Swin.pdf)

\#**Swin**    \#**CV**    \#**backbone**

- 将transformer用作CV领域通用的backbone时，具有以下难点：
  
  1. 在NLP中，一个单词就是一个基本元素，而在CV领域，不同视觉元素的空间尺度可以相差很大，不同尺度的信息提取，对于物体检测等下游任务至关重要
  
  2. 一张图像中的像素数要远远超过一段文字当中的单词数，而自注意力的计算复杂度与图像尺寸的平方成正比，对于语义分割等需要像素级精度的任务，高分辨率图像计算代价难以承受（ViT）

- Swin将图像分割成大小4×4的patch，为了产生具有层次结构的特征表达，采用batch merging层，将输入中2×2的邻接patch（特征维度为C）连接（特征维度4C），并通过一个线性层将特征维度降低到2C，同时分辨率降低为原来的1/2

- 相比ViT计算全局自注意力的方式，Swin采用平移窗口（包含M×M个patch，e.g.,M=7）方案，只在窗口内计算自注意力，计算复杂度与图像尺寸成正比；transformer block交替使用两种窗口划分的自注意力模块，一种是常规划分的W-MSA，另一种SW-MSA通过将常规窗口平移(⌊M/2⌋, ⌊M/2⌋)得到，自注意力可以使用cyclic-shifting和attention masking的方式计算
  
  ![swin_cyclic.png](./images/swin_cyclic.png)

- 计算自注意力时，对每个head引入一个可学习的相对位置偏置矩阵$B∈R^{M^{2}\times M^{2}}$，可以提高模型准确率
  
  $$
  Attention(A,K,V)=SoftMax(QK^T/\sqrt{d}+B)V
  $$

- 使用DropPath而不是Dropout进行正则化

### 模型架构

![swin.png](./images/swin.png)

### pytorch实现

- [github链接](https://github.com/wushidiguo/Swin-Transformer)

- SW-MSA生成mask的部分代码
  
  ```python
  img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
  h_slices = (slice(0, -self.window_size),
              slice(-self.window_size, -self.shift_size),
              slice(-self.shift_size, None))
  w_slices = (slice(0, -self.window_size),
              slice(-self.window_size, -self.shift_size),
              slice(-self.shift_size, None))
  cnt = 0
  for h in h_slices:
      for w in w_slices:
          img_mask[:, h, w, :] = cnt
          cnt += 1
  
  mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
  mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
  attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
  attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
  ```
