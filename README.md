# AgenticRAG 运行方式

### 1. 环境准备

```bash
# 环境 1: agenticrag（推理/评测/数据合成/SFT）
conda create -n agenticrag python=3.11
conda activate agenticrag
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# SFT 训练额外安装：cd LLaMA-Factory && pip install -e .

# 环境 2: verl（GRPO 训练，与 agenticrag 严格隔离）
conda create -n verl python=3.12
conda activate verl
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
cd verl && pip install -e . && cd ..
pip install -r requirements-verl.txt
```

### 2. 模型准备

下载 BGE-M3 和 BGE-Reranker-v2-m3 到 `models/` 目录：
```bash
mkdir -p models
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3
```

### 3. 构建检索索引

```bash
python scripts/build_index.py \
    --corpus data/datasets/corpus.json \
    --index-dir data/indexes
```

### 4. 构建知识图谱

```bash
python scripts/build_knowledge_graph.py \
    --corpus data/datasets/corpus.json \
    --output-graph data/indexes/knowledge_graph.json \
    --output-embeddings data/indexes/entity_embeddings.pkl \
    --triples-cache data/indexes/triples_cache.jsonl \
    --lang zh
```

### 5. 配置模型 API

所有 LLM 调用统一通过 OpenAI 兼容协议（`llm/client.py`），支持本地 vLLM 和任意云端 API。

**本地 vLLM 部署**（推荐）：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --served-model-name Qwen3-4B \
    --port 9097 \
    --gpu-memory-utilization 0.45 \
    --max-model-len 32768
```

**环境变量配置**：
```bash
# 本地 vLLM 地址
export VLLM_BASE_URL="http://localhost:9097/v1"     # Agent 推理模型

# 模型和检索配置
export MODEL_HUB="./models"
export AGENT_LLM_MODEL="Qwen3-4B"
export PROMPT_LANG="zh"

# Judge 模型（评测 + GRPO reward 中的 LLM 评分）
# 方案 A: 本地 vLLM 部署
export JUDGE_BASE_URL="http://localhost:8086/v1"
export JUDGE_LLM_MODEL="gpt-oss-120b"
# 方案 B: 云端 API（推荐）— 在 llm/client.py 中注册模型后使用
# export JUDGE_LLM_MODEL="gpt-4o-judge"
```

**添加自定义模型**（本地或云端均可）：在 `llm/client.py` 的 `MODEL_CONFIGS` 中新增条目：
```python
# 云端 API 示例（如 OpenAI / DeepSeek / 通义千问等）
MODEL_CONFIGS["gpt-4o-judge"] = ModelConfig(
    url="https://api.openai.com/v1",
    model_name="gpt-4o",
    api_key=os.environ.get("OPENAI_API_KEY", ""),
)
# 本地 vLLM 示例
MODEL_CONFIGS["my-local-model"] = ModelConfig(
    url="http://localhost:8000/v1",
    model_name="my-model-name",
)
```

### 6. 运行评测

```bash
export NEWS_CORPUS_DIR=data/financial_eval
export NEWS_INDEX_DIR=data/financial_all/indexes

# Pipeline 模式评测
python scripts/run_cloud_eval.py --model Qwen3-4B --workers 1

# Agentic 模式评测
python scripts/eval_agentic.py --model Qwen3-4B --max-samples 50
```

### 7. 数据合成

```bash
# 1. 生成种子 QA
python scripts/gen_seed_qa.py

# 2. 多跳合成
python scripts/domain_multihop_synthesis.py --lang zh

# 3. 质量过滤
python scripts/judge_synthesis.py
python scripts/clean_synthesis.py
```

### 8. SFT 训练

```bash
# 1. 构建 Oracle Trace → SFT 数据
python scripts/build_oracle_traces.py
python scripts/trace_to_sft.py --lang zh
python scripts/convert_sft_to_llamafactory.py

bash training/sft_pipeline.sh
```

### 9. GRPO 训练

```bash
python scripts/prepare_agentic_grpo_data.py
bash training/start_retrieval_server.sh
export VERL_DIR=$(pwd)/verl
bash training/start_grpo.sh
```
