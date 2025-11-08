# -.get
Get somethings interesting
# 社会计算与演化项目

> 该仓库包含若干用于数据抓取、文本分析、情感分析与演化博弈可视化的 Python 脚本与示例数据。整体设计目标为：清晰、可复现、易于扩展。

---

## 文件概览

* `雪球.py` — 实现对单只个股的**日度行情数据抓取**（从雪球或其他数据源）。
* `微博爬取.py` — 按照关键词抓取微博帖子（文本、作者、时间、转发/评论/点赞数等元信息）。
* `关键词爬取_redbook.zip` — 一个压缩包，包含按关键词抓取小红书笔记的脚本与示例配置（请解压后查看 `README`）。
* `redbook.py` — 小红书自动化评论脚本（用于演示自动化流程，请注意合规与平台规则）。
* `股吧评论分析.py` — 对股吧（论坛）评论进行**预处理 + 情感分析**，并输出汇总统计与可视化图表。
* `greenwash.py` — 基于异质性代理人的**演化博弈仿真结果可视化**脚本（生成演化轨迹、群体分布热图等）。
* `Failed_2025.9.10 微博与男女平权/` — 一个失败的实验案例目录，包含微博爬取与文本分析的最小示例。目标是比较中国互联网上男性主义与女性主义话语权，但因样本过少未得出结论。保留以供复现与方法论反思。

---

## 快速开始

1. 克隆仓库：

```bash
git clone <your-repo-url>
cd <repo-folder>
```

2. 建议使用虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
.\.venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
```

> `requirements.txt` 建议包含：
>
> ```text
> requests
> beautifulsoup4
> lxml
> pandas
> numpy
> tqdm
> matplotlib
> seaborn
> scikit-learn
> jieba
> transformers  # 如果使用预训练语言模型
> selenium      # 如果使用浏览器自动化
> ```

---

## 各脚本使用说明（示例）

### `雪球.py`

用途：抓取并保存单只股票的日度行情（CSV）。

示例：

```bash
python 雪球.py --ticker 600519 --start 2020-01-01 --end 2025-10-31 --out data/600519_daily.csv
```

常见参数：

* `--ticker`：股票代码
* `--start` / `--end`：日期范围
* `--out`：输出路径

### `微博爬取.py`

用途：按关键词抓取微博（支持分页、时间范围、按热度/时间排序）。

示例：

```bash
python 微博爬取.py --keyword "男女平权" --max 2000 --since 2025-01-01 --until 2025-09-30 --out data/weibo_kw.csv
```

注意事项：

* 抓取频率请遵守平台限制并使用合适的 sleep 参数。
* 如果使用 API/cookies 登录，请将凭证放入 `config.yaml` 并加入 `.gitignore`。

### `关键词爬取_redbook.zip`

解压后请阅读其中的 `README`，通常包含：

* `config_example.json`：关键词与代理/并发设置
* `crawl_redbook.py`：主抓取脚本

### `redbook.py`

用途：自动化评论小红书笔记（演示）。

**强烈提醒**：自动化操作可能违反平台服务条款。仅在可控、合规的实验或测试帐号上运行。

示例：

```bash
python redbook.py --note-id 123456789 --comment "感谢分享！" --repeat 5
```

### `股吧评论分析.py`

用途：文本清洗、分词、情感打分、主题或聚类分析，并输出图表/CSV汇总。

示例：

```bash
python "股吧评论分析.py" --input data/guba_comments.csv --output reports/guba_sentiment_summary.csv
```

脚本会生成：

* 评论长度分布图
* 情感得分随时间的折线图
* 高频词词云（需额外安装 `wordcloud`）

### `greenwash.py`

用途：读取仿真结果（或直接运行内置仿真），绘制代理人行为随时间的演化曲线与群体分布热图。

示例（读取结果）：

```bash
python greenwash.py --input results/sim_hetero.npy --plot-dir plots/
```

示例（直接运行仿真并绘图）：

```bash
python greenwash.py --run-sim --agents 500 --steps 200 --hetero-level 0.3
```

输出：

* `trajectory.png`：行为占比随时间变化
* `heatmap.png`：参数扫描的最终分布

---

## 关于失败的案例：`Failed_2025.9.10 微博与男女平权`

该文件夹包含：爬取脚本、数据样本与分析笔记。失败原因分为：

1. **样本量不足**——关键话题抓取时，匹配到的高质量帖子较少，导致统计显著性不足；
2. **筛选偏差**——抓取策略可能集中在部分用户或话题圈层；
3. **指标设计**——话语权难以仅用简单情感分布或点赞数衡量。

建议保留该目录作为复现材料，并在未来的重复实验中改进：扩大时间窗口、改进关键词列表、引入用户网络结构信息、使用更鲁棒的因果识别策略等。

---

## 数据与隐私

* 请勿将任何包含用户个人隐私或敏感信息的数据上传到公共仓库。将凭证、cookies、或原始抓取数据放入 `.gitignore`。
* 对外发布前请对收集到的文本数据做脱敏处理。

---

## 开发建议与扩展方向

* 将抓取逻辑抽象为可复用模块（`crawler` package），并把平台特定代码维持在 `platforms/` 下。
* 为每个抓取脚本添加单元测试与速率控制测试。
* 对情感分析模块封装接口，方便替换不同模型（词典、SVM、BERT-based）。
* 将仿真与可视化分离，支持批量参数扫描并自动产生日志。

---

## 许可证 & 致谢

本仓库默认使用 MIT 许可证（若需其他许可证，请替换 `LICENSE` 文件）。

---

如果你希望把 README 调整成英文版、添加快速演示 GIF、或把其中某个脚本改写为可安装的包（`pip install .`），告诉我你偏好的风格与细节，我会继续帮你完善。
