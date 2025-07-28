# SJT_Agent

<p align="center">
    <img src="./chatarena-text.png" width="500px"/>
</p>

<h3 align="center">
    <p>基于ChatArena的情境判断测试（SJT）多智能体对话平台</p>
</h3>

[![License: Apache2](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/chatarena/chatarena/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/chatarena)](https://pypi.org/project/chatarena/)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

---

SJT_Agent 是一个基于 ChatArena 的情境判断测试（SJT）多智能体对话平台，支持多种大模型后端和自定义环境。

## 在线演示
- [Demo 地址](#)  
- [演示视频](#)

## 安装方法

### 依赖要求
- Python >= 3.7
- OpenAI API Key（可选，用于 GPT-3.5/4）

### 安装步骤

#### 1. 安装 ChatArena 及本项目依赖
```bash
pip install chatarena
pip install -r requirements.txt
```

#### 2. 配置环境变量
在项目根目录下创建 `.env` 文件，内容如下（根据需要填写）：
```
OPENAI_API_KEY=你的OpenAI密钥
SUPABASE_URL=你的Supabase数据库URL（可选）
SUPABASE_SECRET_KEY=你的Supabase密钥（可选）
```

## 使用方法

### 启动Web界面
```bash
python app.py
```

### 启动Flask Web服务
```bash
python web/app.py
```

### 命令行运行SJT实验
```bash
python run_sjt.py
```

## 依赖环境
详见 [requirements.txt](./requirements.txt)

## 贡献方式
欢迎提交 issue 或 pull request！

## 许可证
MIT License

---
