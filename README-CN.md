# 新一代智能在线判题系统

[![Python](https://img.shields.io/badge/python-3.12-blue.svg?style=flat-square)](https://www.python.org/)
[![Django](https://img.shields.io/badge/django-3.2-blue.svg?style=flat-square)](https://www.djangoproject.com/)
[![Vue](https://img.shields.io/badge/vue-2.x-brightgreen.svg?style=flat-square)](https://vuejs.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)

> 基于 Python Django + Vue.js 的智能在线判题系统，深度融合大语言模型、知识图谱与深度学习推荐技术。

[English](README.md)

---

## 项目概述

本项目是一套面向高校师生及编程学习者的新一代智能在线判题（Online Judge）系统。在传统 OJ 功能之上，创新性地引入了 AI 大语言模型多智能体协作、基于 Neo4j 的知识图谱构建与表示学习、以及多模态深度学习推荐引擎，构建了从诊断到推荐再到个性化辅导的完整学习闭环。

### 核心功能

**基础平台**
- 用户注册登录、邮箱激活、JWT Token 安全认证
- 题库管理：多维度筛选、关键字搜索、分页浏览
- 多语言代码在线编写与沙箱安全判题（C / C++ / Java / Python）
- 7 种判题结果：Accepted、Wrong Answer、TLE、MLE、RE、CE 等
- ACM 与 OI 双赛制竞赛系统，支持实时排行榜与赛后复盘

**AI 智能辅助**
- **MasterAgent 多智能体框架**：下设 6 个专业智能体协同工作
  - ProfileAgent — 生成个性化学习画像
  - RecommendAgent — 融合多路召回推荐题目
  - HintAgent — 渐进式解题提示
  - ErrorAnalysisAgent — 提交失败自动诊断
  - PathPlanningAgent — 个性化学习路径规划
  - ResourceAgent — 自动生成教学资源
- 多轮上下文感知对话，大语言模型驱动的智能问答

**深度学习推荐引擎**
- 多路召回 + 精排架构
- **GraphSAGE 异构图神经网络（HeteroGNN）**：利用用户-题目-知识点异构图生成图嵌入
- **Transformer 序列模型**：自注意力建模用户做题轨迹
- **DeepFM 深度因子分解机**：精排阶段建模低阶特征交互与高阶非线性组合

**知识图谱与表示学习**
- Neo4j 图数据库构建知识点 PREREQUISITE_OF 依赖图谱
- **RGCN（关系图卷积网络）** 半监督节点分类
- **TransE 翻译模型**学习三元组嵌入表示
- 可视化交互式知识图谱，已掌握/未掌握状态区分

**学习分析**
- 个人统计概览：提交数、通过率、击败百分比、知识点分布
- 学习趋势图表、多维学习者画像
- 教案管理与 Markdown + 数学公式渲染

**管理后台**
- 题目增删改查、测试用例配置、代码模板管理
- 比赛创建管理、用户权限管理
- 数据看板：用户统计、题目分布、提交趋势可视化

---

## 技术架构

```
┌─────────────────────────────────────────────────┐
│                   前端 (Vue.js)                   │
│     iView / Element UI / ECharts / CodeMirror    │
├─────────────────────────────────────────────────┤
│              后端 API (Django REST)               │
│  account │ problem │ contest │ submission │ judge │
│  aiChat │ agents │ recommend │ knowledge_graph   │
│  lesson_plan │ learning_stats │ dashboard        │
├─────────────────────────────────────────────────┤
│   PostgreSQL │ Redis │ Neo4j │ ChromaDB │ Nginx   │
├─────────────────────────────────────────────────┤
│         AI 服务 (讯飞星火 / DeepSeek)             │
│         判题沙箱 (Seccomp / Docker)               │
│         异步任务队列 (Dramatiq)                    │
└─────────────────────────────────────────────────┘
```

## 环境要求

- Python 3.12+
- PostgreSQL 14+
- Redis 7+
- Neo4j 5.x
- Node.js 16+（前端构建）
- Docker & Docker Compose

## 快速开始

### 1. 克隆仓库

```bash
git clone <本项目地址>
cd <项目目录>
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入数据库连接、AI API Key 等配置
```

### 3. 后端初始化

```bash
pip install -r deploy/requirements.txt
python manage.py migrate
python manage.py initadmin  # 创建初始管理员账号
python manage.py runserver
```

### 4. 前端构建

```bash
cd ../OJFE
npm install
npm run build
# 构建产物输出到 OJ/frontend_dist/
```

### 5. Docker 一键部署（生产环境推荐）

```bash
docker compose up -d
```

---

## 项目结构

```
├── account/           # 用户系统
├── agents/            # AI 多智能体
├── aiChat/            # AI 对话
├── announcement/      # 公告
├── conf/              # 判题服务器配置
├── contest/           # 竞赛系统
├── dashboard/         # 管理后台数据看板
├── feedback/          # 用户反馈
├── judge/             # 判题引擎
├── knowledge_graph/   # 知识图谱
├── learning_stats/    # 学习分析
├── lesson_plan/       # 教案管理
├── oj/                # Django 项目配置
├── options/           # 系统选项
├── problem/           # 题库管理
├── recommend/         # 推荐引擎
├── submission/        # 提交判题
├── utils/             # 工具函数
├── deploy/            # 部署配置
└── frontend_dist/     # 前端构建产物（部署时）
```

---

## 第三方依赖声明

本项目基于以下优秀的开源库构建，特此致谢：

### 后端核心依赖

| 库 | 许可证 | 用途 |
|------|------|------|
| Django 3.2 | BSD | Web 框架 |
| Django REST Framework | BSD | REST API 框架 |
| psycopg2 | LGPL | PostgreSQL 驱动 |
| dramatiq | LGPLv3 | 异步任务队列 |
| django-redis | BSD | Redis 缓存后端 |
| Pillow | HPND | 图像处理 |
| XlsxWriter | BSD | Excel 导出 |
| neo4j-driver | Apache 2.0 | Neo4j 图数据库驱动 |
| chromadb | Apache 2.0 | 向量数据库 |
| openai | Apache 2.0 | LLM API 调用 |
| PyTorch | BSD | 深度学习框架 |
| scikit-learn | BSD | 机器学习 |
| numpy | BSD | 科学计算 |
| gunicorn | MIT | WSGI 服务器 |

### 前端核心依赖

| 库 | 许可证 | 用途 |
|------|------|------|
| Vue.js 2.x | MIT | 前端框架 |
| Vuex | MIT | 状态管理 |
| Vue Router | MIT | 路由管理 |
| Vue I18n | MIT | 国际化 |
| axios | MIT | HTTP 客户端 |
| iView | MIT | UI 组件库 |
| Element UI | MIT | UI 组件库 |
| CodeMirror | MIT | 代码编辑器 |
| KaTeX | MIT | 数学公式渲染 |
| highlight.js | BSD | 代码高亮 |
| ECharts | Apache 2.0 | 数据可视化 |
| moment.js | MIT | 日期处理 |

本项目本身基于 [QingdaoU/OnlineJudge](https://github.com/QingdaoU/OnlineJudge)（MIT 许可证）二次开发，在前者基础上进行了大量扩展与改进，新增了 AI 智能体、深度学习推荐引擎、知识图谱等核心模块。原始版权声明见下方。

---

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

本项目在 [QingdaoU/OnlineJudge](https://github.com/QingdaoU/OnlineJudge)（MIT 许可证）的基础上进行扩展开发，原始代码版权归原作者所有：

```
Copyright (c) 2017-present Qingdao University OnlineJudge Contributors
```

新增模块代码版权归本项目作者所有。
