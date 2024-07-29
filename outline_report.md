# RecSys Models Report

Generally:

- Traditional: FM, MF, ALS
- Deep
  - FC layer, MLP
  - Attention Layer
  - Sequential
  - Cross feature
- Combination
  - e.g. DeepFM, DCN

- Task-wise
  - Retrieval
  - Ranking
  - Sequence pred
  - Multi-task

- Graph-based
  - GNN
  - KG
- KD
- Contrastive learning
- Data augmentation

- also all kinds of variants

- Shared: HPT
- Distributed / parallelism

## EasyRec by Alibaba
### 简介
实现SOTA深度模型，提供简便的config设置以及超参tuning

骨架组件化，灵活搭建模型

案例：
```
model_config: {
  model_name: "WideAndDeep"
  model_class: "RankModel"
  feature_groups: {
    group_name: 'wide'
    feature_names: 'user_id'
    feature_names: 'movie_id'
    feature_names: 'job_id'
    feature_names: 'age'
    feature_names: 'gender'
    feature_names: 'year'
    feature_names: 'genres'
    wide_deep: WIDE
  }
  feature_groups: {
    group_name: 'deep'
    feature_names: 'user_id'
    feature_names: 'movie_id'
    feature_names: 'job_id'
    feature_names: 'age'
    feature_names: 'gender'
    feature_names: 'year'
    feature_names: 'genres'
    wide_deep: DEEP
  }
  backbone {
    blocks {
      name: 'wide'
      inputs {
        feature_group_name: 'wide'
      }
      input_layer {
        only_output_feature_list: true
        wide_output_dim: 1
      }
    }
    blocks {
      name: 'deep_logit'
      inputs {
        feature_group_name: 'deep'
      }
      keras_layer {
        class_name: 'MLP'
        mlp {
          hidden_units: [256, 256, 256, 1]
          use_final_bn: false
          final_activation: 'linear'
        }
      }
    }
    blocks {
      name: 'final_logit'
      inputs {
        block_name: 'wide'
        input_fn: 'lambda x: tf.add_n(x)'
      }
      inputs {
        block_name: 'deep_logit'
      }
      merge_inputs_into_list: true
      keras_layer {
        class_name: 'Add'
      }
    }
    concat_blocks: 'final_logit'
  }
  model_params {
    l2_regularization: 1e-4
  }
  embedding_regularization: 1e-4
}
```

适用任务：
- Candidate Generation（召回）
- Scoring（排序）

测试模型：
- Wide & Deep
![wide_and_deep](EasyRec/docs/images/models/wide_and_deep.png)
    - Wide: Wide **linear** models; cross-product特征变换的**线性**模型
    - Deep: Deep neural networks; 稀疏特征embedding的FNN
    - 结合**记忆**和**泛化**的优势搭建推荐系统
    - 应用场景：Google Play
    - 记忆：
    通过叉乘记忆特征交互，利用历史数据的相关性
    - 泛化：通过低维密集embedding泛化到未出现的特征组合
    - 本质就是一个hybrid（observed + unobserved），同时捕捉specific + general patterns
    - 适用场景：CTR，电商推荐，个人搜索，etc.
- DeepFM
![deepfm](EasyRec/docs/images/models/deepfm.png)
    - Factorization Machines (FM) + DNN
    - FM和DNN共享相同的特征，即相同Embedding
    - FM捕捉低阶特征交互，DNN建模高阶
    - word embedding & Doc2Vec捕捉上下文语义；GRU学习用户兴趣动态变化
    - 适合稀疏数据；容易overfit
    - 适用场景：similar to Wide&Deep，可以结合LSTM设计动态场景下的广告推荐
- DCN（Deep & Cross Network）
![dcn](EasyRec/docs/images/models/dcn.png)
    - Deep: MLP
    - Cross Network:每一层应用特征交叉，模型能自动计算所有可行的特征组合，无需手动进行feature engineering
    - Scalability↑；发现New item
    - 适用场景：similar，MTL（多任务学习）
- AutoInt
![autoint](EasyRec/docs/images/models/autoint.png)
    - Automatic Feature Interaction Learning via **Self-Attentive** Neural Networks
    - 残差连接的多头自注意力网络，低维空间中显式模拟特征交互
    - numerical和categorical特征映射到同一低维空间
    - outperform Wide&Deep 以及 DeepFM
    - 自注意力机制查找预测中最关键的特征组合
    - 类似模型：DLRM(Deep Learning Recommendation Model for Personalization and Recommendation Systems[Facebook])
    - 适用场景：大规模数据，可解释性强
- 共同点：稀疏特征（CTR等），基本推荐预测、排序

其他模型：
- DSSM（Deep Structured Semantic Models）:双塔**召回**
    - 类似模型以及改进：
        - DropoutNet（提供variant）
          - end2end
          - Negative Mining负采样
          - cold start
        - MIND（Multi-Interest Network with Dynamic routing）：加入兴趣聚类，支持多兴趣召回，想法类似CrossNet
- MultiTower：不采用FM，所以embedding可以有不同的dimension

  - 可配合使用模型
    - DIN：建模用户点击序列（sequence）
    - BST：多头自注意力，同样建模sequence

- CL4SRec（Contrastive Learning for Sequential Recommendation）：
![cl4srec](EasyRec/docs/images/models/cl4srec.jpg)
利用对比学习框架从用户行为序列生成自监督信息，提取更有信息量的用户行为进行encode。 另外还采用了三种序列数据增强方式(crop/mask/reorder)构建自监督信息，embedding层以及encoder在三者间是互通的。

    适用场景：用户行为数据

- CDN （Cross Decoupling Network）
![cdn](EasyRec/docs/images/models/cdn.jpg)
item反馈通常是长尾分布，少数item接收了大部分用户的反馈，因此推荐的item是有偏的。CDN通过混合专家结构，在item段解耦记忆和泛化的学习过程
    
    适用场景：有偏分布；平衡item

- CMBF（Cross-Modal-Based Fusion Recommendation Algorithm）
![cmdf](EasyRec/docs/images/models/cmdf.png)
    - 多模态交叉信息，解决稀疏/cold start
    - Visual feature: CNN-based / Transformer-based
    - 相似模型：UNITER（UNiversal Image-TExt Representation Learning，基于Transformer）

- ESMM（Entire Space Multi-task Model）:估计点击后转化率 (CVR)，利用条件概率

- Multi-task models
    - MMoE（Mixture of Multimodal Interaction Experts）:适用于任务之间相关性低的scenario，用于负迁移（negative transfer）现象，即相关性不强的任务之间的信息共享
    - PLE（Progressive Layered Extraction）：解决跷跷板现象(seesaw phenomenon)，即提升一部分任务的效果，同时牺牲另外部分任务的效果

- PDN（Path-based Deep Network）
    - Trigger Net + SimNet，捕获user对其每个交互item的兴趣 + 评估每个交互item与目标item之间的相似度（user -> trigger item -> target item）
- PPNet（Parameter Personalized Net）
  - 设计出 gating 机制，增加网络参数个性化并能够让模型快速收敛
  - 适用场景：个性化推荐

- CoMetricLearningI2I
  - 基于session点击数据计算item与item的相似度
  - 保证dist(``anchor_item``, ``pos_item``) 远小于 dist(``anchor_item``, ``neg_item``)

- Rocket Launching
  - 解决类似CTR对响应时间要求严格的场景
  - 轻量light net + 助推器复杂网络booster net
  - 本质就是个KD

- FiBiNet
  - SENET(Squeeze-Excitation network)
    - 动态学习特征的重要性
  - Bilinear Feature Interaction
    - 特征交叉，结合Inner Product & Hadamard Product

- etc.

总结：
  - 召回
    - DSSM
    - MIND
    - CoMetricLearningI2I
    - PDN
    - DropoutNet
  - 排序
    - MultiTower
    - DeepFM
    - FM
    - WideAndDeep
    - DLRM
    - DCN
    - AutoInt
    - DIN
    - BST
    - Rocket Launching
    - MaskNet
    - FiBiNet
    - Cross Decoupling Network
    - PPNet
    - CL4SRec
    - 多模态
      - Highway Network
      - CMBF
      - UNITER
  - 多目标
    - ESMM
    - MMoE
    - DBMTL
    - AITM
    - PLE


### 配置
- 训练
    - optimizer（支持多优化器）
    - train_distribute：PS-Worker模式和 All-Reduce模式
    - num_steps (= total_sample_num * num_epochs / batch_size / num_workers)
    - ckpt路径设置

- 评估
```
eval_config {
  metrics_set: {
    auc {}
  }
  metrics_set: {
    accuracy {}
  }
  metrics_set: {
    gauc {} # grouped auc
  }
}
```

- 模型
    - [Wide & Deep组件化配置示例](#简介)
    - [DeepFM示例](EasyRec/docs/source/models/deepfm.md#2-组件化模型)
    - [DCN示例](EasyRec/docs/source/models/dcn.md#dcn-v1-配置说明)
    - [Autoint示例](EasyRec/docs/source/models/autoint.md#配置说明)

### API使用小结
- scikit-learn包兼容性问题，Python需>= 3.9，并找到对应的tensorflow version
- 训练过程：TBD（w/o CUDA driver）

## TensorFlow Recommenders

### 简介

该项目提供完整的流程：数据准备、模型制定、训练、评估和部署。它基于Keras构建，希望得到平缓的学习曲线，同时也提供构建复杂模型的灵活性。

[API doc](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/)提供较为详细的reference。

### 模型

- 多任务Multitask：
    - 针对≥2个目标进行优化，因此有≥2个loss
        - 设置权重
        ```
        model = MovielensModel(rating_weight=1.0, retrieval_weight=1.0)
        ```
    - 任务之间共享变量，实现transfer learning
    - e.g. MovieLens 预测评分 + 预测是否看电影

- Sequential recommendation:
    - 查看user之前交互过的一系列item，预测下一item
    - 包含其他特征信息，e.g. MovieLens：年份，评分，类型
    - Pipeline：
      - 预处理：定义架构 for parsing
      - 序列模型：双塔
        - query tower：编码sequence w/ GRU
        - candidate tower：candidate movie
    - 适用场景：包含时序信息的预测

- DCN（Deep & Cross Network）
  - 提供``tfrs.layers.dcn.Cross``构建cross层，deep层使用``tf.keras.layers.Dense``即可
  - 模型架构：embedding层 -> CrossNet -> DeepNet
  - ``tfrs.layers.dcn.Cross``一些重要参数：
    - ``projection_dim``：用于减少计算成本【低秩``input_dim`` -> ``projection_dim`` < ``input_dim/2``】
    ![low-rank DCN](tf_recommenders/assets/low_rank.png)
    - ``kernel_initializer``：内核矩阵initializer
  - 其他Variants：
    - DCN w/ parallel structure
    - Cross layers concatenated

### API使用小结

基本Pipeline（standard）:
  - 定义模型（Both User & Item）
  - 定义任务（例如：召回，并指明metric）
  - 训练拟合并eval，使用``model.compile``，``model.fit``，``model.evaluate``

TensorFlow Recommenders提供任务接口，e.g. ``tfrs.tasks.Retrieval``指定召回任务，该对象内置地将loss和metric计算捆绑在一起；同时提供metric接口，e.g. ``tfrs.metrics.FactorizedTopK``计算前k个candidate的指标。

e.g.
```
tfrs.tasks.Ranking(
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[
        tf.keras.metrics.RootMeanSquaredError("RMSE")
      ]
    )
```
另外，还可以使用``tf.keras.layers.Embedding``快速构建embedding，这里一般添加额外的embedding来解释未知token。

预测结果部署：
  - 导出预测层（e.g. ``BruteForce`` layer），保存索引
  - 提供user id，返回预测前k位
  - Alternative：也可用``ScaNN``做Approximate Nearest，准确度↓，速度↑↑（[Efficient serving](recommenders/docs/examples/efficient_serving.ipynb)）

任务：基础任务包括Ranking, Retrieval, Multi-task, etc.

## TorchRec

### 简介
  - Parallelism：大型、高性能多设备/多节点模型，支持混合数据并行/模型并行
  - TorchRec分片器使用不同的分片策略对embedding表进行分片
  - 设备间传输，设备间通信以及计算进行数据重叠，提高性能

[API doc](https://pytorch.org/torchrec/) provides detailed reference.
- ``torchrec.distributed``实现模型并发
- 提供对[FBGEMM](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu)内核的抽象，包括批处理embedding包、数据布局转换和量化支持等
- ``DistributedModelParallel``（DMP）帮助模型分片

### 模型

- Bert4Rec
  - 序列推荐，双向编码器（BERT）
  - 并行同时体现在
    - 数据并行（Transformer）
    - 模型并行（Embedding分布）

- DLRM
  - Pipelined
    - 数据加载
    - 数据并行和模型并行通信
    - 前/后向传播

适用场景：双塔；稀疏稠密混合情境（numerical & categorical）；分布式模型，大型高并发


### API使用小结
TBD（V100 or A100 required）

## RecStudio

### 简介

适用场景：
- 通用推荐
- 序列推荐
- 基于知识推荐
- 基于特征推荐
- 社交推荐

该项目同样模块化模型设计，目前只在召回模型上有实现。

### 模型

官网提供了全部的[model list](http://recstudio.org.cn/docs/models_and_datasets/model_list/)，以下列出了代表性的模型种类：
  - Retrievers
    - General Rec
      - 各种各样的MF
    - Sequential Rec
      - GRU
      - CoSeRec：contrastive self-supervised learning
        - 利用augmentation展现item间correlation
    - Graph-based Rec
      - e.g. GNN-based
      - 在user-item图结构上传播embedding
      - 显式地在user-item图中展现高阶连通性
    - KG-based Rec
      - user-item交互以三元组形式纳入KG
      - 在图谱上CF，e.g. 使用TransE学习user和item的embedding
  - Rankers
    - CTR

数据集种类：
  - TripletDataset：矩阵分解模型（BPR, NCF）
  - UserDataset：自编码器模型（MultiVAE, RecVAE）
  - SeqDataset：序列化推荐模型（GRU4Rec, SASRec）
  - Seq2SeqDataset：序列化推荐掩码预测（Bert4Rec）
  - ALSDataset：交替优化系列模型（WRMF）

类似于EasyRec，只需手动编辑config文件配置模型

### API使用小结

TBD（NVIDIA driver not loaded）


## Recommenders

### 简介

基本Pipeline：
  - 准备数据
  - 建模
  - 评估
  - 优化：调参
  - 运营

样例中提供PySpark帮助并行计算，提高效率

### 模型

原repo的readme提供了[完整表格](recommenders/README.md#algorithms)。

- MF：ranking
  - ALS：大型数据集显式或隐式反馈的MF，针对可扩展性和分布式计算能力进行优化，PySpark环境中运行。
    - 提供超参tuning
    - pyspark提供API：``pyspark.ml.recommendation.ALS``
    - e.g.
    ```
      als = ALS(
      rank=10,
      maxIter=15,
      implicitPrefs=False,
      regParam=0.05,
      coldStartStrategy='drop',
      nonnegative=False,
      seed=42,
      **header
    )
    ```
  - BPR（Bayesian Personalized Ranking）
    - 适合Implicit feedback
    - pairwise loss（observed样本比unobserved排名高）
    - 使用``cornac.models.BPR``
    - e.g.
    ```
      bpr = cornac.models.BPR(
      k=NUM_FACTORS,
      max_iter=NUM_EPOCHS,
      learning_rate=0.01,
      lambda_reg=0.001,
      verbose=True,
      seed=SEED
    )
    ```
  - RLRMC
    - Riemann共轭梯度优化的MF，较小的内存消耗
    - 使用``recommenders.models.rlrmc.RLRMCalgorithm``
  - SVD
    - ``surprise``库作为推荐模组，使用``surprise.SVD``即可
    - e.g.
    ```
      svd = surprise.SVD(random_state=0, n_factors=200, n_epochs=30, verbose=True)
    ```


- Sequential Models
  - SLI-Rec（Short-term and Long-term preference Integrated
RECommender system）
    - 捕获精确推荐系统长期（CF）和短期（RNN）的用户偏好
    - Attentive Asymmetry-SVD范式进行长期建模
    - 改进LSTM中的一些gate，同时考虑时间和语义的不规则性
    - API call：``recommenders.models.deeprec.models.sequential.sli_rec``
  - GRU
    - both short-term & long-term
  - Caser（Convolutional Sequence Embedding Recommendation）
    - CNN
    - 同时捕捉单点级别+联合级别pattern（sequential info）
  - NextItNet
    - dilated CNN
  - SUM
    - 多channel网络结构，相当于MHA，表示user兴趣的不同方面
  - SASRec（Self-Attentive Sequential Recommendation）
    - Transformer-based
    - 所需参数：
      - num_items
      - 用户交互历史最大长度
      - Transformer模块数量
      - embed_dimension（item）
      - attn_dim
      - num_attn_heads
      - dropout rate
      - 卷积层dims
      - L2正则参数
  - SSPET（Sequential Recommendation Via Personalized Transformer）
    - 与SASRec类似

  - Deep Models
    - xDeepFM
      - 同时捕捉低阶和高阶特征交互
        - CIN：显式学习特征交互 + 传统DNN：隐式

    - LightGCN
      - simplified GCN
        - 适用场景：图网络数据
        - 编码embedding中的交互信号
  
  - 其他模型
    - BiVAE
      - 适用于CF任务
      - 为二元数据定制的VAE，具有对称性，更容易从user和item两侧扩展辅助数据
      - 其他使用情景：文档-单词矩阵；例如联合聚类
      - e.g.
      ```
      bivae = cornac.models.BiVAECF(
        k=LATENT_DIM,
        encoder_structure=ENCODER_DIMS,
        act_fn=ACT_FUNC,
        likelihood=LIKELIHOOD, # 优化的似然函数
        n_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        use_gpu=torch.cuda.is_available(),
        verbose=True
      )
      ```

    - Multinomial VAE
      - 与BiVAE类似
      - 使用多项似然的生成模型

    - LightFM
      - FM for both explicit & implicit
    
    - LightGBM 
      - 梯度提升树，content-based快速训练和低内存

  - 新闻推荐
    - LSTUR
      - 同时捕捉长短期偏好
      - 类似于SLI-Rec

    - DKN
      - 适用场景：新闻推荐（其他模型：LSTUR）
      - content-based
      - D结合KG信息，使用TransX方法（e.g. TransE）进行知识图表示学习，应用CNN框架结合实体embedding与词embedding
      - CTR预测过一个基于注意力的神经打分器
      - 使用注意力模块来动态计算用户的聚合历史表征
      - ``recommenders.models.deeprec.models.dkn``





### API使用小结

  - config在创建模型类时编辑，例如：
  ```
  als = ALS(
    rank=10,
    maxIter=15,
    implicitPrefs=False,
    regParam=0.05,
    coldStartStrategy='drop',
    nonnegative=False,
    seed=42,
    **header
` )
  ```


## spotlight

### 简介

基于PyTorch构建深层/浅层推荐模型，提供大量loss fn，表示（浅层MF表示、深层sequence模型）和用于获取或生成数据集的module。

### 模型

- 序列模型
  - Implicit sequence：``spotlight.sequence.implicit.ImplicitSequenceModel``
    - loss：pointwise, bpr, hinge
    - 表示：pooling, cnn, lstm, mixture
      - 提供序列表示对象：``spotlight.sequence.representations``

- 分解模型
  - Implicit
  - Explicit

总结：模型太老，可选太少，最多用作demo/教学

### API小结

测试使用Python版本：3.6