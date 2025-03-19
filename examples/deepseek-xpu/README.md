# DeepSeek-R1 671B Distributed Inference Guide

This guide provides detailed instructions on how to run inference for **DeepSeek-R1 671B** in a distributed setting using **SGLang** and **VLLM**. The model is deployed on **Intel Max GPUs**, leveraging multiple startup methods to optimize performance.

---

## 🔍 Background Information
We have explored and validated multiple methods for running **DeepSeek-R1 671B** on Intel Max GPUs. The two primary approaches include:
- **SGLang + VLLM**: A hybrid approach utilizing both SGLang and VLLM to enhance distributed inference.
- **VLLM**: Running the model directly with VLLM.

This guide focuses on setting up and executing the **SGLang + VLLM** method for optimized inference.

---

## ⚙️ Environment Setup
### 📌 Prerequisites
Ensure that you have the following system dependencies installed:
- Python 3.10+
- Conda (recommended for managing dependencies)
- Intel Max GPU drivers (latest version)

### 🛠️ Python Dependencies
Install the required Python packages:
```bash
pip install setuptools==75.6.0 packaging==24.2 msgspec blake3 py-cpuinfo compressed_tensors gguf partial_json_parser
conda install libsqlite=3.48.0
```

### 🔧 Triton Setup
You need the latest **Triton** compiler and a compatible **PyTorch (2.7)** build. Install the latest Triton package from Intel’s nightly builds:
```bash
pip install triton-3.2.0+git6fa2562b-cp310-cp310-linux_x86_64.whl
```
Triton builds can be found here: [Intel Triton Nightly Builds](https://github.com/intel/intel-xpu-backend-for-triton/actions/workflows/nightly-wheels.yml)

### 🔥 PyTorch Setup
**Torch 2.7** disables `xccl` by default, so you need to rebuild it with `xccl` enabled.
```bash
# Clone Triton source
TRITON_COMMIT=6fa2562b
git clone https://github.com/intel/intel-xpu-backend-for-triton.git
cd triton && git checkout ${TRITON_COMMIT}

# Clone and patch PyTorch
TORCH_COMMIT=c21dc11
git clone https://github.com/pytorch/pytorch.git
cd pytorch && git checkout ${TORCH_COMMIT}
git apply /path/to/triton/scripts/pytorch_fp64.patch

# Build PyTorch with xccl
USE_XCCL=ON USE_STATIC_MKL=1 python setup.py bdist_wheel
```

### 📦 VLLM Setup
Due to partial support issues on Intel Max GPUs, we use a customized branch of **VLLM**:
```bash
git clone --branch yuhua/deepseek --single-branch https://github.com/zhuyuhua-v/vllm.git
cd vllm
pip install setuptools_scm
pip install setuptools==75.6.0 packaging==24.2
VLLM_TARGET_DEVICE=xpu python setup.py install
```

### 🚀 SGLang Setup
Similarly, a specific branch of **SGLang** is required for compatibility:
```bash
git clone --branch deepseek-xpu-distribute --single-branch https://github.com/cboss6/sglang.git
cd sglang
pip install -e "python[all_xpu]"
```

---

## 🏃‍♂️ Running DeepSeek-R1 671B
### 📂 Offline Inference
#### 🖥️ Single-Tile Execution
Run the model with a batch size of **1**, input length **32**, and output length **32**:
```bash
python3 -m sglang.bench_one_batch --batch-size 1 --input 32 --output 32 \
    --model deepseek-ai/DeepSeek-R1 --trust-remote-code --device xpu
```

#### 🏗️ Multi-Tile Execution with Tensor Parallelism
For distributed inference across **8 tiles**, use:
```bash
python3 -m sglang.bench_one_batch --batch-size 1 --input 1024 --output 32 \
    --tp-size 8 --model deepseek-ai/DeepSeek-R1 --trust-remote-code --device xpu
```

#### 📊 Profiling
To measure inference performance, enable profiling:
```bash
python3 -m sglang.bench_one_batch --batch-size 1 --input 1024 --output 32 \
    --tp-size 8 --model deepseek-ai/DeepSeek-R1 --trust-remote-code --device xpu --profile
```

### 🌐 Serving Inference via API
#### 🖥️ Launching the Server
Start a model-serving instance:
```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --trust-remote-code --device xpu
```

#### 🔗 Sending Requests (Client-Side)
Use the OpenAI-compatible API interface to send inference requests:
📌 Refer to the official guide: [SGLang Benchmarking - DeepSeek](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-sending-requests-with-openai-api)

---

## 📌 Additional Notes
- For optimal performance, ensure that your **Intel Max GPU drivers** and **"OneAPI** are up to date.
- If you encounter any **XPU compatibility issues**, consider adjusting **batch sizes**, **tensor parallelism**, or using **alternative kernel configurations**.

---

🔥 **Happy Experimenting with DeepSeek-R1 671B with Intel GPU!** 🚀
