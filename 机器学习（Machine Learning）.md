# 机器学习（Machine Learning）

## 1. 什么是机器学习

**机器学习（Machine Learning, ML）** 是人工智能（Artificial Intelligence, AI）的一个重要分支。它让计算机从数据中学习规律，并利用学到的规律对新数据进行预测或决策，而不需要人为写出处理每一种情况的固定规则。

机器学习的核心可以概括为：

> 数据（Data）→ 特征（Features）→ 模型（Model）→ 预测（Prediction）→ 评估与改进（Evaluation and Improvement）

一个机器学习任务通常包含以下要素：

- **样本（Sample / Instance）**：一条数据，例如一名学生、一张图片或一封邮件。
- **特征（Feature）**：描述样本的属性，例如年龄、身高、像素或词频。
- **标签（Label / Target）**：希望模型预测的结果，例如“是否患病”或“房价”。
- **模型（Model）**：从输入特征到输出结果的映射关系。
- **参数（Parameter）**：模型通过训练从数据中学到的数值，例如线性回归中的权重。
- **超参数（Hyperparameter）**：训练前人为设置的参数，例如 KNN 中的 $K$、决策树的最大深度和学习率。

---

## 2. 机器学习的主要类型

### 2.1 监督学习（Supervised Learning）

监督学习使用带有正确标签的数据进行训练，目标是学习输入 $X$ 与标签 $y$ 之间的关系。

主要任务：

1. **分类（Classification）**：预测离散类别。
   - 例：垃圾邮件识别、手写数字识别、疾病诊断。
   - 常用算法：逻辑回归（Logistic Regression）、K 近邻（K-Nearest Neighbors）、决策树（Decision Tree）、支持向量机（Support Vector Machine）、朴素贝叶斯（Naive Bayes）。

2. **回归（Regression）**：预测连续数值。
   - 例：房价预测、气温预测、销量预测。
   - 常用算法：线性回归（Linear Regression）、决策树回归（Decision Tree Regression）、随机森林回归（Random Forest Regression）。

### 2.2 无监督学习（Unsupervised Learning）

无监督学习处理没有标签的数据，目标是发现数据中隐藏的结构或规律。

主要任务：

- **聚类（Clustering）**：把相似样本归为一组。常用算法有 K-Means、层次聚类（Hierarchical Clustering）和 DBSCAN。
- **降维（Dimensionality Reduction）**：减少特征数量，同时尽量保留重要信息。常用算法有主成分分析（Principal Component Analysis, PCA）、t-SNE 和 UMAP。
- **异常检测（Anomaly Detection）**：识别与大多数数据明显不同的样本，例如欺诈交易或设备故障。
- **关联规则（Association Rules）**：发现不同事物之间的共现关系，例如购物篮分析。

### 2.3 强化学习（Reinforcement Learning, RL）

强化学习让智能体（Agent）与环境（Environment）不断交互，并根据奖励（Reward）学习最佳策略（Policy）。

基本概念：

- **智能体（Agent）**：执行动作的主体。
- **环境（Environment）**：智能体所处的外部世界。
- **状态（State）**：当前环境信息。
- **动作（Action）**：智能体可以进行的操作。
- **奖励（Reward）**：环境对动作的反馈。
- **策略（Policy）**：智能体在不同状态下选择动作的方法。
- **价值函数（Value Function）**：衡量某个状态或动作长期收益的函数。

强化学习的目标不是只追求眼前奖励，而是最大化**累计奖励（Cumulative Reward）**。典型应用包括机器人控制、自动驾驶、游戏 AI 和资源调度。

### 2.4 其他学习方式

- **半监督学习（Semi-supervised Learning）**：使用少量有标签数据和大量无标签数据。
- **自监督学习（Self-supervised Learning）**：从数据本身构造训练目标，是现代大语言模型和视觉模型的重要训练方式。
- **迁移学习（Transfer Learning）**：把一个任务中学到的知识迁移到另一个相关任务。

---

## 3. 标准机器学习流程（Machine Learning Workflow）

### 3.1 明确问题（Problem Definition）

首先确定任务类型、输入、输出和成功标准。例如，“根据房屋信息预测价格”属于回归任务，“识别邮件是否为垃圾邮件”属于二分类任务（Binary Classification）。

### 3.2 收集与理解数据（Data Collection and Understanding）

数据质量通常决定模型效果的上限。需要检查：

- 数据量是否足够；
- 标签是否准确；
- 数据是否具有代表性；
- 是否存在缺失值（Missing Values）、重复值或异常值（Outliers）；
- 数据中是否包含偏见（Bias）或隐私信息。

### 3.3 数据预处理（Data Preprocessing）

常见处理包括：

- 缺失值填充或删除；
- 类别特征编码，例如独热编码（One-Hot Encoding）；
- 特征缩放：标准化（Standardization）或归一化（Normalization）；
- 文本分词、图像缩放等特定数据处理；
- 处理类别不平衡（Class Imbalance），例如重采样或设置类别权重。

标准化常用公式：

$$
z=\frac{x-\mu}{\sigma}
$$

其中 $\mu$ 是均值，$\sigma$ 是标准差。KNN、SVM 等依赖距离的算法通常特别需要特征缩放。

### 3.4 划分数据集（Dataset Split）

- **训练集（Training Set）**：用于学习模型参数。
- **验证集（Validation Set）**：用于选择模型和调整超参数。
- **测试集（Test Set）**：只在最后使用，用于估计模型面对新数据时的真实表现。

常见比例是 70%/15%/15% 或 80%/10%/10%。数据较少时，可使用 **K 折交叉验证（K-Fold Cross-Validation）**。

必须避免**数据泄漏（Data Leakage）**：测试集的信息不能以任何方式参与训练或预处理参数的计算。

### 3.5 特征工程（Feature Engineering）

特征工程是从原始数据中创建、选择或转换更有用的特征，例如：

- 从日期提取年、月、星期；
- 将身高和体重组合成 BMI；
- 使用特征选择（Feature Selection）删除无关或高度冗余的特征；
- 使用 PCA 进行特征提取（Feature Extraction）。

### 3.6 训练与调参（Training and Hyperparameter Tuning）

训练的本质是通过优化算法减小**损失函数（Loss Function）**。常见优化方法是梯度下降（Gradient Descent）：

$$
\theta \leftarrow \theta-\eta\nabla_\theta L(\theta)
$$

其中 $\theta$ 是模型参数，$\eta$ 是学习率（Learning Rate），$L$ 是损失函数。

超参数搜索方法包括：

- 网格搜索（Grid Search）；
- 随机搜索（Random Search）；
- 贝叶斯优化（Bayesian Optimization）。

### 3.7 评估、部署与监控（Evaluation, Deployment and Monitoring）

模型上线（Deployment）后仍需监控数据分布和性能。现实数据发生变化称为**数据漂移（Data Drift）**；输入与目标之间的关系发生变化称为**概念漂移（Concept Drift）**。出现漂移时可能需要重新训练模型。

---

## 4. K 近邻算法（K-Nearest Neighbors, KNN）

KNN 是一种简单、直观的监督学习算法，可用于分类和回归。它没有显式的训练过程，因此常被称为**惰性学习（Lazy Learning）**。

### 4.1 基本思想

对于一个新样本：

1. 计算它与训练集中所有样本的距离；
2. 找到距离最近的 $K$ 个邻居；
3. 分类任务采用多数投票（Majority Voting）；
4. 回归任务采用邻居目标值的平均值或加权平均值。

### 4.2 常用距离

**欧氏距离（Euclidean Distance）**：

$$
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$

**曼哈顿距离（Manhattan Distance）**：

$$
d(x,y)=\sum_{i=1}^{n}|x_i-y_i|
$$

**切比雪夫距离（Chebyshev Distance）**：

$$
d(x,y)=\max_i|x_i-y_i|
$$

### 4.3 $K$ 值的影响

- $K$ 太小：模型容易受噪声影响，可能发生过拟合。
- $K$ 太大：决策边界过于平滑，可能发生欠拟合。
- 二分类时常选择奇数 $K$，减少票数相同的情况。
- 最合适的 $K$ 通常通过交叉验证确定。

### 4.4 优点与缺点

优点：原理简单、容易实现、无需复杂训练、适合小规模数据。

缺点：预测时计算量大、内存占用较高、对特征尺度和无关特征敏感，在高维空间中效果容易下降，这种现象称为**维度灾难（Curse of Dimensionality）**。

---

## 5. 重点监督学习算法（复习核心）

### 5.1 线性回归（Linear Regression）

#### 5.1.1 基本思想

线性回归用于预测**连续值（Continuous Value）**。它假设目标变量可以由若干特征的线性组合近似表示：

$$
\hat y=w_0+w_1x_1+w_2x_2+\cdots+w_px_p=\mathbf{w}^T\mathbf{x}+b
$$

- $\hat y$：模型预测值；
- $x_j$：第 $j$ 个特征；
- $w_j$：第 $j$ 个特征的回归系数（Coefficient）；
- $b$ 或 $w_0$：截距（Intercept）。

在其他条件不变时，$w_j$ 表示 $x_j$ 每增加 1 个单位，预测值平均改变 $w_j$ 个单位。系数的正负表示影响方向，但**相关性不等于因果关系（Correlation does not imply Causation）**。

#### 5.1.2 损失函数与求解

普通最小二乘法（Ordinary Least Squares, OLS）通过最小化残差平方和来求参数。单个样本的残差为：

$$
e_i=y_i-\hat y_i
$$

均方误差（Mean Squared Error, MSE）为：

$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

参数可通过两种方式求解：

1. **正规方程（Normal Equation）**

$$
\hat{\mathbf w}=(X^TX)^{-1}X^Ty
$$

无需迭代，但特征很多时求逆成本高；$X^TX$ 不可逆时可使用伪逆（Pseudo-inverse）。

2. **梯度下降（Gradient Descent）**

不断沿损失函数下降最快的方向更新参数，更适合大规模和高维数据。学习率（Learning Rate）过大可能震荡或发散，过小则收敛很慢。

#### 5.1.3 主要假设

经典线性回归常见假设：

- **线性关系（Linearity）**：特征与目标的关系可用线性形式描述；
- **误差独立（Independence）**：各样本误差相互独立；
- **同方差性（Homoscedasticity）**：误差方差基本稳定；
- **误差近似正态（Normality of Errors）**：主要影响置信区间和显著性检验；
- **无严重多重共线性（No Severe Multicollinearity）**：特征之间不能高度线性相关。

可通过残差图（Residual Plot）检查非线性和异方差，通过方差膨胀因子（Variance Inflation Factor, VIF）检查多重共线性。

> 注意：预测任务不一定要求所有统计推断假设都严格成立，但违反假设可能影响稳定性、解释性和预测表现。

#### 5.1.4 正则化线性回归

正则化（Regularization）在损失函数中加入对参数大小的惩罚，以减少过拟合。

**岭回归（Ridge Regression / L2 Regularization）**：

$$
L=MSE+\lambda\sum_{j=1}^{p}w_j^2
$$

- 将系数压缩到接近 0，但通常不会变成 0；
- 对多重共线性有效；
- 所有特征通常仍保留。

**Lasso 回归（Lasso Regression / L1 Regularization）**：

$$
L=MSE+\lambda\sum_{j=1}^{p}|w_j|
$$

- 可将部分系数压缩为 0，因此具有特征选择作用；
- 高度相关的特征中可能只保留一个，结果有时不稳定。

**弹性网络（Elastic Net）** 同时使用 L1 和 L2，兼顾特征选择与系数稳定性。

$\lambda$ 越大，惩罚越强，模型越简单；过大会造成欠拟合。使用正则化前一般要标准化特征，截距通常不参加惩罚。

#### 5.1.5 优缺点与复习要点

优点：训练快、结果容易解释、可作为优秀的基线模型（Baseline Model）。

缺点：难以直接表达复杂非线性关系；对异常值敏感；强共线性会导致系数不稳定。

**高频易错点：**

- 线性回归预测连续值；逻辑回归主要做分类。
- “线性”是指对参数 $w$ 线性，不代表输入只能使用原始的一次项；加入 $x^2$ 后仍可构造线性参数模型。
- $R^2$ 高不等于模型一定正确，也不等于存在因果关系。

---

### 5.2 逻辑回归（Logistic Regression）

#### 5.2.1 基本思想

逻辑回归名字中虽然有“回归”，但主要用于**分类（Classification）**。二分类时，它先计算线性得分：

$$
z=\mathbf w^T\mathbf x+b
$$

再通过 Sigmoid 函数把任意实数压缩到 $(0,1)$，得到属于正类的概率：

$$
P(y=1\mid x)=\sigma(z)=\frac{1}{1+e^{-z}}
$$

如果概率大于阈值（Threshold，常见默认值为 0.5），预测为正类，否则预测为负类。阈值不是固定真理：降低阈值通常提高 Recall、降低 Precision；提高阈值通常提高 Precision、降低 Recall。

#### 5.2.2 对数几率与系数解释

几率（Odds）定义为事件发生概率与不发生概率之比：

$$
Odds=\frac{p}{1-p}
$$

逻辑回归假设**对数几率（Log-Odds / Logit）**与特征呈线性关系：

$$
\log\frac{p}{1-p}=\mathbf w^T\mathbf x+b
$$

当其他变量不变时，$x_j$ 每增加 1 单位，Odds 乘以 $e^{w_j}$。因此 $e^{w_j}$ 称为**优势比（Odds Ratio）**。

#### 5.2.3 损失函数

逻辑回归不能直接使用普通线性回归的 MSE 作为标准训练目标，通常用二元交叉熵（Binary Cross-Entropy），也叫对数损失（Log Loss）：

$$
L=-\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
$$

它来自最大似然估计（Maximum Likelihood Estimation, MLE）。模型对“非常自信但预测错误”的样本惩罚很大。

#### 5.2.4 多分类与决策边界

- **一对其余（One-vs-Rest, OvR）**：每次训练“某一类 vs 其他所有类”的二分类器。
- **Softmax Regression / Multinomial Logistic Regression**：一次计算所有类别概率，概率总和为 1。

普通逻辑回归形成线性决策边界（Linear Decision Boundary）。通过多项式特征、交互项或核方法可表达更复杂的边界。

#### 5.2.5 正则化、优缺点与注意事项

逻辑回归同样常使用 L1 或 L2 正则化。正则化强度要通过验证集或交叉验证选择。不同软件中的参数定义可能相反，例如 scikit-learn 的 $C$ 越小，正则化越强。

优点：训练和预测快；能输出概率；可解释性较强；适合高维稀疏数据，例如文本分类。

缺点：默认决策边界是线性的；对异常值、多重共线性较敏感；复杂关系需要特征工程。

**高频易错点：**

- 模型输出的是概率估计，分类结果还取决于阈值。
- 类别不平衡时不能只看 Accuracy，应关注 Precision、Recall、F1、PR-AUC 或 ROC-AUC。
- 逻辑回归中“特征与概率线性”是错误说法；线性的是特征与 Log-Odds。

---

### 5.3 决策树（Decision Tree）

#### 5.3.1 基本思想

决策树通过一系列“如果……那么……”的规则递归划分数据空间：

- **根节点（Root Node）**：包含全部训练样本；
- **内部节点（Internal Node）**：对某个特征进行判断；
- **分支（Branch）**：判断结果；
- **叶节点（Leaf Node）**：输出类别或预测值。

训练目标是选择合适的特征和切分点，使划分后的子节点尽可能“纯”。

#### 5.3.2 分类树的划分指标

**熵（Entropy）**：

$$
H(D)=-\sum_{k=1}^{K}p_k\log_2p_k
$$

样本全部属于同一类时熵为 0；类别越混杂，熵越大。

**信息增益（Information Gain）**：

$$
Gain(D,A)=H(D)-\sum_v\frac{|D_v|}{|D|}H(D_v)
$$

ID3 选择信息增益最大的特征，但容易偏向取值种类多的特征。

**信息增益率（Gain Ratio）**：C4.5 使用信息增益率，减弱对多取值特征的偏好。

**基尼不纯度（Gini Impurity）**：

$$
Gini(D)=1-\sum_{k=1}^{K}p_k^2
$$

CART 分类树常选择能使加权 Gini 最小的切分。Gini 和 Entropy 的实际结果通常相近，Gini 计算略快。

#### 5.3.3 回归树

回归树的叶节点通常输出该叶中训练样本目标值的平均数。常用切分标准是最小化平方误差（Squared Error）或绝对误差（Absolute Error）。因此回归树的预测是分段常数（Piecewise Constant）。

#### 5.3.4 停止生长与剪枝

完全生长的树很容易记住噪声并过拟合。控制方法包括：

- `max_depth`：最大深度；
- `min_samples_split`：内部节点继续划分所需的最少样本数；
- `min_samples_leaf`：叶节点至少包含的样本数；
- `max_leaf_nodes`：最大叶节点数；
- `max_features`：每次划分考虑的特征数；
- `ccp_alpha`：代价复杂度剪枝（Cost-Complexity Pruning）的强度。

**预剪枝（Pre-pruning）** 在建树过程中提前停止；**后剪枝（Post-pruning）** 先生成较完整的树，再删除收益较小的分支。

#### 5.3.5 优缺点与易错点

优点：规则直观；可处理非线性关系和特征交互；通常不要求特征标准化；分类树和回归树都可使用。

缺点：单棵树方差高，数据稍有变化就可能得到不同结构；深树容易过拟合；贪心划分不能保证全局最优。

**高频易错点：**

- 决策树一般不依赖欧氏距离，因此通常不需要归一化或标准化。
- 决策树可自动学习非线性边界，但单棵树并不稳定。
- 特征重要性（Feature Importance）不等于因果影响；基于不纯度的重要性还可能偏向连续或高基数特征。

---

### 5.4 集成学习（Ensemble Learning）

集成学习把多个基学习器（Base Learner / Weak Learner）组合起来。其直觉是“多个模型共同决策”通常比单个模型更稳定，但基模型必须具有一定准确性和差异性（Diversity）。

#### 5.4.1 Bagging

**Bagging（Bootstrap Aggregating）** 的流程：

1. 从原始训练集进行有放回抽样（Bootstrap Sampling），生成多个训练子集；
2. 每个子集独立、并行训练一个基模型；
3. 分类采用投票，回归采用平均。

Bagging 主要降低**方差（Variance）**，特别适合决策树这类高方差模型。

##### 随机森林（Random Forest）

随机森林在 Bagging 的基础上增加**随机特征子集（Random Feature Subspace）**：每个节点切分时只考虑部分特征。这会降低树之间的相关性，使集成效果更好。

关键参数：

- `n_estimators`：树的数量，增加后通常更稳定，但训练和预测更慢；
- `max_depth`、`min_samples_leaf`：控制每棵树的复杂度；
- `max_features`：节点切分时考虑多少特征；
- `class_weight`：处理类别不平衡；
- `bootstrap`：是否使用有放回抽样。

未被某棵树抽中的样本称为**袋外样本（Out-of-Bag Samples, OOB）**，可以用于袋外评估（OOB Score），无需额外验证集即可得到近似泛化误差。

优点：稳健、能处理非线性和特征交互、预处理要求较少、可并行训练。

缺点：模型体积和计算量大于单棵树；整体解释性下降；回归时通常难以外推到训练标签范围之外。

#### 5.4.2 Boosting

Boosting 按顺序训练多个弱学习器，后一个模型重点修正前面模型的错误，最后加权组合。它通常更侧重降低**偏差（Bias）**，也可能降低方差，但对噪声和参数设置更敏感。

##### AdaBoost

AdaBoost（Adaptive Boosting）提高被前一轮错误分类样本的权重，让后续模型更关注难分样本。最终结果是多个弱分类器的加权投票。它对错误标签和异常值较敏感。

##### GBDT

梯度提升决策树（Gradient Boosting Decision Tree, GBDT）逐轮拟合损失函数的**负梯度（Negative Gradient）**。在平方误差回归中，可以直观理解为后续树不断拟合前一轮的残差：

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

- $F_{m-1}(x)$：当前集成模型；
- $h_m(x)$：新加入的弱学习器；
- $\eta$：学习率（Learning Rate）。

`learning_rate` 与 `n_estimators` 存在权衡：较小学习率通常需要更多树，但往往有更好的泛化能力。树太深、轮数太多可能过拟合，可使用早停（Early Stopping）。

##### XGBoost、LightGBM 与 CatBoost

- **XGBoost（Extreme Gradient Boosting）**：在 GBDT 基础上加入二阶梯度、显式正则化、列采样和缺失值处理，工程实现成熟。
- **LightGBM（Light Gradient Boosting Machine）**：使用直方图算法和叶子优先生长（Leaf-wise Growth），训练快、内存效率高；小数据上需限制叶节点，避免过拟合。
- **CatBoost**：擅长处理类别特征（Categorical Features），通过有序目标统计等方法降低目标泄漏风险。

#### 5.4.3 Voting 与 Stacking

**投票法（Voting）**：

- 硬投票（Hard Voting）：选择票数最多的类别；
- 软投票（Soft Voting）：对各模型的预测概率加权平均，要求概率最好经过校准（Calibration）。

**Stacking（堆叠泛化）**：使用多个基础模型的预测作为新特征，再训练一个元学习器（Meta-learner）完成最终预测。

训练 Stacking 时必须使用**折外预测（Out-of-Fold Prediction, OOF）**训练元学习器；如果直接使用基础模型对其训练数据的预测，会发生数据泄漏和严重过拟合。

#### 5.4.4 Bagging 与 Boosting 对比

| 对比项 | Bagging | Boosting |
|---|---|---|
| 训练方式 | 基模型相互独立，可并行 | 基模型前后依赖，通常串行 |
| 数据处理 | Bootstrap 有放回抽样 | 关注错分样本或拟合负梯度 |
| 主要作用 | 降低方差 | 主要降低偏差，也可降低方差 |
| 典型算法 | Random Forest | AdaBoost、GBDT、XGBoost、LightGBM |
| 对噪声敏感度 | 相对稳健 | 通常更敏感 |
| 调参难度 | 相对较低 | 相对较高 |

**一句话速记：Bagging 是“独立训练后平均”，Boosting 是“按顺序纠错”。**

---

### 5.5 其他常用算法速记

#### 5.5.1 支持向量机（Support Vector Machine, SVM）

SVM 寻找能使两类样本间隔（Margin）最大的超平面。距离边界最近、决定边界位置的样本称为**支持向量（Support Vectors）**。

- $C$ 较大：更重视训练错误，间隔可能变窄，过拟合风险增加；
- $C$ 较小：允许更多训练错误，间隔更宽，正则化更强；
- 核技巧（Kernel Trick）可在不显式映射高维特征的情况下处理非线性，常见核包括线性核和径向基核（Radial Basis Function, RBF）；
- RBF 的 $\gamma$ 越大，单个样本影响范围越小，边界更复杂；$\gamma$ 越小，边界更平滑。

SVM 对特征尺度敏感，通常需要标准化；适合中小规模、高维数据，但大规模数据训练成本较高。

#### 5.5.2 朴素贝叶斯（Naive Bayes）

根据贝叶斯定理：

$$
P(y\mid x)=\frac{P(x\mid y)P(y)}{P(x)}
$$

朴素贝叶斯假设给定类别后各特征条件独立。该假设现实中往往不完全成立，但模型仍常有良好效果。

- Gaussian Naive Bayes：连续特征近似服从高斯分布；
- Multinomial Naive Bayes：常用于词频、计数特征；
- Bernoulli Naive Bayes：常用于二值特征。

优点是速度快、对小数据友好，常用于文本分类；缺点是条件独立假设较强，特征相关性高时性能可能下降。

#### 5.5.3 K-Means 聚类

K-Means 是无监督聚类算法，目标是最小化样本到所属簇中心的平方距离：

$$
J=\sum_{k=1}^{K}\sum_{x_i\in C_k}\|x_i-\mu_k\|^2
$$

步骤为“初始化中心 → 分配样本 → 更新中心 → 重复直到收敛”。它对初始中心、异常值和特征尺度敏感，因此通常需要标准化并使用 K-Means++ 初始化。可用肘部法（Elbow Method）或轮廓系数（Silhouette Score）辅助选择 $K$。

---

### 5.6 算法横向对比

| 算法 | 任务 | 是否需缩放 | 非线性能力 | 可解释性 | 典型风险 |
|---|---|---:|---:|---:|---|
| Linear Regression | 回归 | 正则化时建议 | 低，需构造特征 | 高 | 异常值、共线性、欠拟合 |
| Logistic Regression | 分类 | 建议 | 默认较低 | 高 | 非线性关系拟合不足 |
| KNN | 分类、回归 | **需要** | 较强 | 中 | 维度灾难、预测慢 |
| Decision Tree | 分类、回归 | 通常不需要 | 强 | 单树较高 | 高方差、过拟合 |
| Random Forest | 分类、回归 | 通常不需要 | 强 | 中 | 模型较大、解释性下降 |
| GBDT / XGBoost | 分类、回归 | 通常不需要 | 强 | 中 | 调参复杂、可能过拟合 |
| SVM | 分类、回归 | **需要** | 使用核函数时强 | 较低 | 大数据训练慢、参数敏感 |
| Naive Bayes | 分类 | 视分布而定 | 有限 | 中 | 条件独立假设较强 |

#### 选择算法的复习思路

- 需要简单、快速、可解释的回归基线：Linear Regression。
- 需要概率输出和可解释的分类基线：Logistic Regression。
- 数据量较小且边界复杂：可尝试 KNN 或 SVM。
- 表格数据、存在非线性和特征交互：优先尝试树模型及其集成。
- 追求稳健、少调参：Random Forest。
- 追求表格数据预测性能且可以认真调参：GBDT、XGBoost、LightGBM 或 CatBoost。
- 文本词频等高维稀疏数据：Logistic Regression、Linear SVM 或 Naive Bayes。

> “最佳算法”不能仅凭名称决定，必须通过验证集或交叉验证，在相同数据划分和评估指标下公平比较。

---

## 6. 模型评估指标（Evaluation Metrics）

### 6.1 分类指标

混淆矩阵（Confusion Matrix）包含：

- **TP（True Positive）**：预测为正，实际为正；
- **TN（True Negative）**：预测为负，实际为负；
- **FP（False Positive）**：预测为正，实际为负；
- **FN（False Negative）**：预测为负，实际为正。

常用指标：

$$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
$$

$$
Precision=\frac{TP}{TP+FP}
$$

$$
Recall=\frac{TP}{TP+FN}
$$

$$
F1=2\times\frac{Precision\times Recall}{Precision+Recall}
$$

- **准确率（Accuracy）**：整体预测正确的比例；类别严重不平衡时可能具有误导性。
- **精确率（Precision）**：预测为正的样本中有多少确实为正，适合关注误报（False Positive）的任务。
- **召回率（Recall / Sensitivity）**：实际为正的样本中有多少被找出，适合关注漏报（False Negative）的任务。
- **F1 Score**：Precision 和 Recall 的调和平均。
- **ROC-AUC**：衡量模型在不同阈值下区分正负样本的能力。

### 6.2 回归指标

- **平均绝对误差（Mean Absolute Error, MAE）**：误差绝对值的平均，对异常值相对不敏感。
- **均方误差（Mean Squared Error, MSE）**：误差平方的平均，对大误差惩罚更强。
- **均方根误差（Root Mean Squared Error, RMSE）**：MSE 的平方根，与目标变量单位相同。
- **决定系数（Coefficient of Determination, $R^2$）**：衡量模型解释目标变化的程度；数值越接近 1 通常越好，但也可能小于 0。

指标应根据业务代价选择。例如疾病筛查更重视 Recall，垃圾邮件过滤则需要考虑误把正常邮件判为垃圾邮件的代价。

---

## 7. 过拟合、欠拟合与泛化

- **过拟合（Overfitting）**：训练集表现很好，但新数据表现差。模型记住了训练数据中的噪声。
- **欠拟合（Underfitting）**：模型过于简单，在训练集和测试集上都表现不好。
- **泛化能力（Generalization）**：模型在未见数据上的表现能力，是机器学习真正关心的目标。

常见改进方法：

- 收集更多或更有代表性的数据；
- 使用交叉验证；
- 简化过于复杂的模型；
- 使用正则化（Regularization），如 L1 或 L2；
- 进行特征选择；
- 神经网络中使用 Dropout、数据增强（Data Augmentation）和早停（Early Stopping）。

**偏差—方差权衡（Bias-Variance Trade-off）**：简单模型往往偏差高、方差低；复杂模型往往偏差低、方差高。建模需要在二者之间寻找平衡。

---

## 8. 深度学习与机器学习的关系

**深度学习（Deep Learning, DL）** 是机器学习的一个子领域，主要使用多层神经网络自动学习特征。

- **卷积神经网络（Convolutional Neural Network, CNN）**：常用于图像处理。
- **循环神经网络（Recurrent Neural Network, RNN）**：适合序列数据，但长距离依赖处理较困难。
- **Transformer**：通过注意力机制（Attention Mechanism）处理序列，是现代大语言模型（Large Language Model, LLM）的核心结构。

传统机器学习通常更适合数据量较小、结构化的表格数据，并且更容易解释；深度学习通常需要更多数据和计算资源，但在图像、语音和自然语言等非结构化数据上具有强大能力。

---

## 9. 必须牢记的要点

1. 高质量数据通常比盲目更换复杂算法更重要。
2. 测试集只能用于最终评估，必须防止数据泄漏。
3. 训练分数高不代表模型真正有效，要关注泛化能力。
4. 不存在适合所有问题的最佳算法，应结合数据、目标和成本选择。
5. 评估指标必须与实际业务目标一致。
6. 机器学习模型可能继承数据中的偏见，需要考虑公平性（Fairness）、隐私（Privacy）、可解释性（Interpretability）和安全性（Safety）。
7. 建立简单的基线模型（Baseline Model）后，再逐步增加复杂度并记录实验结果。

## 10. 一句话总结

机器学习就是让计算机利用数据学习可泛化的规律；完整过程不仅是选择算法，还包括问题定义、数据处理、特征工程、训练、评估、部署和持续监控。
