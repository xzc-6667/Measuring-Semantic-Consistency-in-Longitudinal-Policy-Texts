# Measuring Semantic Consistency in Longitudinal Policy Texts

Code and materials for the manuscript *Measuring Semantic Consistency in Longitudinal Policy Texts:
A Dynamic Anchor Approach to Central–Local Attention Allocation*.

The procedure is referred to in the paper as **DASM** (Dynamic Anchor Semantic Measurement). The
scripts in `Data and Code/` implement the measurement pipeline described in Section 3, produce the
results reported in Section 4, and reproduce the baselines used in Sections 3.1.2 and 4.3.

## What is here

The scripts are research code, not a packaged tool. They were written and run one stage at a time
during the analysis, and several of them were used to generate a specific figure or table rather
than to form part of a single callable pipeline. File paths and model paths are set as constants near
the top of each script and point to the directory layout of the machine on which the analysis was
carried out; they need to be edited before a script will run elsewhere. Intermediate files
(per-year keyword lists, sentence-level embedding matrices, similarity matrices, semantic cluster
JSON files) are read and written between stages and are not bundled here.

- **Chinese file names.** Most scripts carry Chinese file names because they were named after the
  analysis step they perform. The table below gives the English equivalent of each.
- **Encoding.** All scripts and text files are UTF-8. Several scripts read Chinese corpora and use
  `jieba` for segmentation.
- **Encoders.** The reported results use `bert-base-chinese`. BGE, E5 and DeepSeek-R1 are included
  only for the encoder comparison in Section 3.1.2 and are not part of the measurement pipeline.
  DeepSeek-R1 is called through a local Ollama endpoint (`http://127.0.0.1:11434/api/generate`).

## Files

### Corpus preparation

| File | Step |
|---|---|
| `分句子.py` (sentence splitting) | Splits a policy document into sentences on Chinese sentence-final punctuation (。！？；) and writes the result to Excel. Used to build the sentence-level corpus that all later stages consume. |
| `统计可视化/数据集字数.py` (dataset size) | Counts characters per document across the corpus years. Produces the corpus size statistics reported in Section 3.1.1. |

### Stage 1 — keyword extraction and semantic networks (Section 3.3)

| File | Step |
|---|---|
| `第一步：TF-IDF每年关键词前8.py` (TF-IDF top-8 keywords per year) | Extracts the top-ranked keywords for each year with TF-IDF, after applying the Harbin Institute of Technology stopword list and the 106 domain stopwords listed in Appendix A. |
| `第二步_语义网络.py` (semantic network) | Builds the co-occurrence network for each policy stage, weights edges by PMI, extracts core signals, and writes the phase-level semantic clusters (with weights) to JSON. Also holds the five-stage division used throughout (2004–07, 2008–12, 2013–17, 2018–22, 2023–26) and the Chinese–English term mapping used for the figures. |
| `语义网络表_en.py` (network table, English) | Computes the topological properties reported for each stage (nodes, edges, density, modularity, average path length, top-degree hubs). Produces the stage-level network table in Section 4.1. |

### Stage 2 — matching provincial documents to anchors (Section 3.4)

| File | Step |
|---|---|
| `阈值设定.py` (threshold setting) | Embeds every sentence once and then evaluates the full range of similarity thresholds against the expert-annotated benchmark, producing the threshold optimisation curve used in Section 4.3.1. |
| `综合热力图.py` (combined heatmap) | Builds the five-sentence context window (target sentence plus two preceding and two following), embeds the windows with BERT, matches provincial text against the stage anchors, and assembles the stage-by-year similarity matrix. |
| `江苏省/三位指标待语义权重.py` (three indicators with semantic weights) | Computes responsiveness, persistence, and foresight for each province from the weighted anchor clusters. |
| `江苏省/第三步_四川江苏两省合并_en.py` (merge Sichuan and Jiangsu, English) | Merges the two provincial matrices and produces the combined output used for the cross-province comparison. |
| `江苏省/江苏省热力图.py` (Jiangsu heatmap) | Produces the Jiangsu-only similarity matrix and figure. |

### Stage 3 — indices and figures (Sections 3.4.2 and 4.2)

| File | Step |
|---|---|
| `表 3 川苏两省政策注意力配置.py` (Table 3, attention allocation in Sichuan and Jiangsu) | Computes per-stage responsiveness (R), the Path Dependence Index (P/R) and the Policy Exploration Index (F/R) for both provinces. Reproduces Table 3. |
| `两省注意力配置.py` (attention allocation, two provinces) | Box and line plots of the attention indicators by province and stage. |
| `两省热力图最终版_en.py` (final two-province heatmap, English) | Final version of the combined heatmap figure. |

### Encoder comparison (Section 3.1.2)

| File | Step |
|---|---|
| `输入选择比较/bert模型.py` | Scores the benchmark with BERT-base-chinese (cosine similarity of mean-pooled embeddings). |
| `输入选择比较/bge-base-zh-v1.5.py` | Same, with BGE-base-zh-v1.5. |
| `输入选择比较/intfloatmultilingual-e5-large.py` | Same, with multilingual E5-large. |
| `输入选择比较/cs_deepseek.py` | Scores the benchmark with DeepSeek-R1-14B through the Ollama API, using the prompt given in Appendix B.2 (English) and B.3 (Chinese original). Temperature 0.0; the A–E rubric is normalised to [0, 1]. |
| `模型比较箱线图.py` (model comparison boxplot) | Score distributions across the four encoders. |
| `折线模型图比较.py` (model comparison line plot) | Temporal mean curves across the four encoders. Together with the boxplot script this produces Figure 3. |

### Stopword validation (Section 3.2)

| File | Step |
|---|---|
| `输入选择比较/停用词比较.py` (stopword comparison) | Compares the HIT-only baseline against the HIT list plus the 106 domain stopwords, on the expert-annotated benchmark. |
| `输入选择比较/停用词比较升级.py` (stopword comparison, revised) | Revised version of the same comparison with the full-sample paired test. |
| `停用词比较可视化.py` (stopword comparison visualisation) | Figure 4 and the per-year gain statistics (19 of 23 years). |

### Baseline models (Section 4.3.2)

| File | Step |
|---|---|
| `基线模型比较/LDA模型.py` | LDA baseline: topic estimation on the pooled corpus, document–topic similarity, and the resulting matrix. |
| `基线模型比较/bert-topic.py` | BERTopic baseline under the same protocol. |
| `基线模型比较/参数敏感性分析代码.py` | Parameter sensitivity for the baseline models (topic count, and the equivalent settings carried over to DASM). |

### Unrelated

| File | Note |
|---|---|
| `临时.py` (temporary) | Scraper for September–November 2025 corn futures prices. Unrelated to the analysis in this paper; left in place because it was in the working directory. Safe to ignore or delete. |

## Order of execution

The stages are sequential; each writes files that the next one reads.

1. `分句子.py` → sentence-level corpus
2. `第一步：TF-IDF每年关键词前8.py` → per-year keyword lists
3. `第二步_语义网络.py` → phase-level semantic clusters (JSON)
4. `语义网络表_en.py` → stage network table
5. `阈值设定.py` → similarity threshold
6. `综合热力图.py` → stage × year similarity matrix
7. `江苏省/三位指标待语义权重.py` → three indicators
8. `表 3 川苏两省政策注意力配置.py` → PDI and PEI
9. `两省热力图最终版_en.py`, `两省注意力配置.py` → figures
10. Baseline and comparison scripts are independent of steps 1–9 and can be run at any point after
    step 1.

## Domain stopwords

The 106 domain stopwords listed in Appendix A are defined inside the scripts that use them
(`第一步：TF-IDF每年关键词前8.py`, `第二步_语义网络.py`, `语义网络表_en.py`, and the three baseline
scripts), in a single `domain_stopwords` set. Terms that appear more than once in the source listing
are counted once, which gives the 106 unique terms reported in the paper.

## Citation

If you use this code, please cite the manuscript above.

## Licence

Code is released for academic reuse. The corpus consists of publicly issued government documents;
please observe the terms of the original sources when redistributing them.
