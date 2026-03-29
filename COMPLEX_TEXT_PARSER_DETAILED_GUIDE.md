# 复杂文本解析模块详细介绍文档（最新版）

## 1. 文档目标

本文档用于说明当前项目中最新版图推理模块的实现细节，覆盖：

- 代码结构框架
- 核心能力与设计意图
- 各阶段实现逻辑
- 分割任务适配方式
- 与旧版思路的差异
- 已完成与待优化项

分析对象以 lib/graph_reasoning_chain_v2.py 为主，并结合 lib/backbone.py 与 lib/rs_vocabulary.py 说明调用链。

---

## 2. 模块定位与总体思路

当前模块是一个端到端联合系统，不是单独文本解析器：

文本结构解析 -> 结构化文本图构建 -> 多尺度视觉候选 -> 双图推理 -> 动态门控过滤 -> 分割特征精炼

目标是提升复杂嵌套描述下的目标定位与分割质量，重点解决：

1. 多实体嵌套语义对齐
2. 无关对象干扰抑制
3. 推理步数与文本复杂度匹配
4. 推理结果与分割任务闭环融合

---

## 3. 代码结构框架

### 3.1 文件级结构

lib/graph_reasoning_chain_v2.py 由 4 个核心类组成：

1. FineGrainedTextParser
2. GraphAttentionLayer
3. GraphConvLayer
4. GraphReasoningChain

职责划分：

- FineGrainedTextParser：复杂文本解析、关系三元组提取、子表达式推理单元生成
- GraphAttentionLayer / GraphConvLayer：图上消息传递基础算子
- GraphReasoningChain：多模态图构建、动态门控推理、双图融合、分割精炼输出

### 3.2 外部依赖关系

- 词表来源：lib/rs_vocabulary.py
- 主干接入：lib/backbone.py 中导入 GraphReasoningChain

调用链：

backbone -> GraphReasoningChain.forward -> FineGrainedTextParser.parse_batch_detailed -> 双图多步推理 -> 分割精炼

---

## 4. FineGrainedTextParser 详细分析

### 4.1 初始化逻辑

构造参数：

- tokenizer_name：BERT tokenizer 名称
- dataset_name：数据集名（用于选择词表）

初始化行为：

1. 尝试加载 BertTokenizer
2. 从 get_lexicon_for_dataset(dataset_name) 加载 entity/attribute/relation 词表
3. 构建两组关系补充词：
   - reference_relations
   - modification_relations

说明：

- 本版不再维护 extended_relations 字段，而是通过 _find_relations_with_type 分类型检索 reference/modification/spatial。

### 4.2 解析接口

本版主接口是批量解析：

- parse_batch_detailed(input_ids, l_mask) -> List[Dict]

保留了兼容单样本接口：

- parse_detailed(...) 仅返回 parse_batch_detailed 的第一个样本

这意味着“批量解析仅首样本有效”的旧限制已修复。

### 4.3 输出结构（单样本）

每个样本返回字段包括：

- target
- entities
- attributes
- relations
- entity_attributes
- spatial_triplets
- entity_tokens_map / attr_tokens_map / rel_tokens_map
- reasoning_units
- valid_len

其中 reasoning_units 是本版新增关键字段，用于后续自适应多步推理。

### 4.4 核心子函数

1. _find_components

- 通用短语匹配器，用于实体与属性
- 支持多词短语
- 使用 replace("##", "") 处理 BERT 子词

2. _find_relations_with_type

- 分三类识别关系词：reference/modification/spatial
- 输出 (relation_word, token_span, relation_type)

3. _build_entity_attribute_map

- 采用“最近实体”启发式分配属性
- 输出 {entity: [attr1, attr2, ...]}

4. _extract_spatial_triplets

- 构建 (reference, relation, target) 结构
- 新增 rel_token_span 供关系节点池化使用

5. _build_reasoning_units

- 根据实体锚点及邻近属性/关系，生成渐进式子表达式区间
- 为自适应推理链提供 step query

---

## 5. 图推理基础算子

### 5.1 GraphAttentionLayer

实现标准图注意力：

- Q/K/V 线性投影
- 缩放点积注意力
- 通过 adjacency mask 限制可传播边

### 5.2 GraphConvLayer

实现归一化邻接聚合：

- 邻接归一化求和
- 线性变换 + LayerNorm

两者在每个推理步串联使用，形成稳定的图消息传递。

---

## 6. GraphReasoningChain 端到端实现

### 6.1 初始化结构（新版）

主要参数：

- in_channels
- text_dim=768
- hidden_dim=256
- num_visual_nodes=64
- num_reasoning_steps=2
- num_steps（兼容旧调用名）

兼容性说明：

- 若传入 num_steps，会覆盖 num_reasoning_steps
- 与 backbone 当前调用保持兼容，无需改调用代码

模块组成：

1. 特征投影

- visual_proj
- text_proj

2. 文本节点投影

- entity_proj
- attr_proj
- rel_proj

3. 主图推理分支

- gat_layers + gcn_layers

4. 关系边扩展与文本调边（新增）

- num_relation_types=14（11 地理关系 + 3 尺度关系）
- rel_gate（目标引导关系权重）
- rel_text_gate（关系文本调边）
- edge_update_alpha（步进式边更新强度）

5. 多尺度候选与噪声抑制（新增）

- multiscale_strides=(1,2,4)
- noise_prior_head（复杂背景噪声抑制）

6. 语义相似图分支（新增）

- cls_gat_layers + cls_gcn_layers

7. 双图融合门（新增）

- graph_fuse

8. 分割适配精炼头（新增）

- seg_refine_head

9. 重构与门控

- reconstruct
- alpha
- refine_alpha

### 6.2 文本图构建（结构化边已落地）

函数：_build_text_graph_single

节点构造顺序：

1. 目标实体节点（固定索引 0）
2. 参考实体节点
3. 属性节点（由 entity_attributes 指导）
4. 关系节点（由 spatial_triplets 与 rel_token_span 指导）

邻接构造策略：

1. 自环
2. 目标 <-> 参考实体
3. 实体 <-> 属性
4. 参考 <-> 关系 <-> 目标 三元组链
5. 弱连接先验（0.05）防止图断裂

结论：旧版“文本子图全连接”的限制已经替换为结构化连边。

### 6.3 视觉图构建（已升级）

函数：_build_visual_adjacency

依据节点空间坐标与尺度等级生成 14 类关系邻接：

- 8 类方位关系：east/south/west/north/se/sw/ne/nw
- 3 类地理拓扑关系：contain/inside/adjacent
- 3 类尺度关系：larger/smaller/similar

关系边权由两路信息共同决定：

- 目标实体引导（rel_gate(target_emb)）
- 关系文本调制（rel_text_gate(rel_text_summary)）

并在每一步推理中继续按子表达式进行边更新（A_vis_t）。

### 6.4 子表达式推理链（新增）

函数：_build_step_queries

逻辑：

1. 读取 parsing 阶段生成的 reasoning_units
2. 从文本特征中池化每个 unit 得到 step query
3. 推理步数 T = min(max_steps, len(reasoning_units))

意义：推理步数随文本复杂度自适应，避免固定步数过短/过长。

### 6.5 动态门控（新增）

函数：_dynamic_gate

逻辑：

1. 节点与当前 step query 做相似度打分
2. 用均值阈值筛选高相关节点
3. 构造 node gate 与 edge gate（外积）

作用：按推理步动态过滤噪声节点与噪声边。

此外，本版加入遥感背景噪声先验（noise_prior_head），在视觉候选打分时对云层/阴影等高噪声区域进行抑制。

### 6.6 双图融合（新增）

图 1：主多模态图 A_main

- 视觉空间边
- 文本结构化边
- 视觉-文本交互边

图 2：语义相似图 A_cls

- 基于节点特征归一化余弦相似度构建

每一步先在两图分别做 GAT+GCN，再用 graph_fuse 自适应融合。

### 6.7 分割任务适配精炼（新增）

不是框回归头，而是分割特征精炼：

1. 将精化后的稀疏视觉节点增量注入回密集特征图（_inject_node_updates）
2. seg_refine_head 生成空间 gate
3. delta = reconstruct(vis_refined)
4. 输出 x + tanh(alpha) * delta * (1 + tanh(refine_alpha) * seg_gate)

这保证模块直接服务分割任务，不引入检测任务依赖。

---

## 7. 关键张量维度说明

主要符号：

- B：batch size
- D：hidden_dim
- K：视觉节点数（<= num_visual_nodes）
- M：文本节点数（动态）
- N：文本 token 数

关键张量：

- x: [B, C, H, W]
- vis_map_all: [B, D, H, W]
- noise_all: [B, 1, H, W]
- vis_all: [B, HW, D]
- txt_all: [B, N, D]
- cand_nodes: [1, K_all, D]（多尺度候选集合）
- text_nodes: [1, M, D]
- vis_nodes: [1, K, D]
- nodes: [1, K+M, D]
- A_main/A_cls: [1, K+M, K+M]

前向计算简式：

1. 视觉/文本投影
2. 批量文本解析（含 reasoning_units）
3. 构建结构化文本图 + 多尺度视觉候选
4. 构建双图邻接
5. 按关系文本调边 + step query 动态门控做多步推理
6. 双图融合
7. 节点增量注入回写密集特征并做分割精炼输出

---

## 8. 当前版本已完成优化点

1. 批量解析

- 已完成 parse_batch_detailed

2. 结构化边

- 已完成实体-属性、参考-关系-目标链连边

2.1 视觉关系扩展

- 已完成 14 类视觉关系边（方位 + 拓扑 + 尺度）

2.2 文本关系调边

- 已完成关系文本驱动的边权调制与步进更新

3. 动态门控

- 已完成节点/边按步过滤

4. 子表达式推理链

- 已完成 reasoning_units -> step queries -> adaptive T

5. 双图融合

- 已完成主图 + 相似图并行推理与可学习融合

6. 分割适配

- 已完成 seg_refine_head，不依赖检测框回归

7. 多尺度与噪声抑制

- 已完成多尺度候选构建与噪声先验抑制

---

## 9. 当前限制与下一步建议

### 9.1 当前限制

1. 文本解析仍以启发式规则为主

- 尚未引入可学习分解监督（例如表达式分解损失）

2. 批内前向仍按样本循环

- 逻辑清晰但吞吐不如全批并行图构建

3. 门控阈值为固定策略

- 目前使用相对均值阈值，尚未自适应学习阈值

4. 语义相似图为特征相似驱动

- 尚未加入类别先验或显式类别图监督

### 9.2 建议优化方向

1. 训练层面

- 增加分解一致性损失与门控稀疏正则

2. 计算效率

- 将单样本图构建改为批量并行

3. 结构先验

- 引入类别图先验，增强同类目标消歧

4. 门控学习

- 将阈值从固定规则改为可学习参数

5. 地理实值特征增强

- 在数据具备条件时，加入真实经纬度/面积等 geo feature 作为节点先验

---

## 10. 与项目的接入方式

### 10.1 接入点

lib/backbone.py 中使用：

from .graph_reasoning_chain_v2 import GraphReasoningChain

并在构造时传入 num_steps。当前模块已兼容 num_steps 与 num_reasoning_steps。

### 10.2 词表配置

lib/rs_vocabulary.py 提供：

- 数据集专用词表
- 默认词表回退

模块会根据 dataset_name 自动选择词表。

---

## 11. 小结

最新版模块已经从“粗粒度文本融合”升级为“结构化文本解析 + 动态门控 + 子表达式推理链 + 双图融合 + 分割精炼”的完整框架。

相较旧版，当前实现已经落地了你提出改造路径中的关键阶段，并且能直接接入本项目分割任务流水线。

在后续迭代中，重点建议继续加强可学习分解监督与批量并行效率，以进一步提升复杂嵌套文本场景下的性能上限。
