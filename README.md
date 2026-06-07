# Next-Generation Intelligent Online Judge

[![Python](https://img.shields.io/badge/python-3.12-blue.svg?style=flat-square)](https://www.python.org/)
[![Django](https://img.shields.io/badge/django-3.2-blue.svg?style=flat-square)](https://www.djangoproject.com/)
[![Vue](https://img.shields.io/badge/vue-2.x-brightgreen.svg?style=flat-square)](https://vuejs.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)

> An intelligent online judge system based on Python Django + Vue.js, deeply integrating LLM, knowledge graphs, and deep learning recommendation.

[中文文档](README-CN.md)

---

## Overview

This project is a next-generation online judge system designed for university students and programming learners. Building upon traditional OJ capabilities, it innovatively introduces LLM-powered multi-agent collaboration, Neo4j-based knowledge graph construction and representation learning, and a multi-modal deep learning recommendation engine — forming a complete learning loop from diagnosis to recommendation to personalized tutoring.

### Core Features

**Core Platform**
- User registration & login with email activation and JWT authentication
- Problem browsing with multi-dimensional filtering, keyword search, and pagination
- Multi-language code editor with sandbox judging (C / C++ / Java / Python)
- 7 judgment statuses: Accepted, Wrong Answer, TLE, MLE, RE, CE, etc.
- ACM and OI contest modes with real-time rankings and post-contest review

**AI-Powered Assistance**
- **MasterAgent multi-agent framework** with 6 specialized agents:
  - ProfileAgent — generates personalized learning profiles
  - RecommendAgent — fuses multi-channel recall for problem recommendation
  - HintAgent — progressive problem-solving hints (direction → key concepts → step-by-step → pitfalls)
  - ErrorAnalysisAgent — automatic diagnosis of WA/TLE/RE failures
  - PathPlanningAgent — personalized learning path planning
  - ResourceAgent — generates lecture notes, quizzes, and coding exercises
- Multi-turn context-aware dialogue powered by LLM

**Deep Learning Recommendation Engine**
- Multi-channel recall + re-ranking architecture
- **GraphSAGE Heterogeneous Graph Neural Network (HeteroGNN)** — leverages user-problem-topic heterogeneous graph for embedding-based recall
- **Transformer sequence model** — self-attention modeling of user problem-solving trajectories
- **DeepFM** — factorized machine for re-ranking, modeling both low-order feature interactions and high-order non-linear combinations

**Knowledge Graph & Representation Learning**
- Neo4j graph database for PREREQUISITE_OF dependency relationships between topics
- **RGCN (Relational Graph Convolutional Network)** for semi-supervised node classification
- **TransE** translation model for triple embedding learning
- Interactive visualized knowledge graph with mastery status indication

**Learning Analytics**
- Personal statistics dashboard: submissions, acceptance rate, percentile ranking, topic distribution
- Learning trend charts and multi-dimensional learner profiles
- Lesson plans with Markdown and LaTeX math rendering

**Admin Dashboard**
- Problem CRUD with test case configuration and code template management
- Contest creation and user role management
- Data dashboard: user statistics, problem distribution, submission trends

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 Frontend (Vue.js)                │
│     iView / Element UI / ECharts / CodeMirror    │
├─────────────────────────────────────────────────┤
│            Backend API (Django REST)             │
│  account │ problem │ contest │ submission │ judge │
│  aiChat │ agents │ recommend │ knowledge_graph   │
│  lesson_plan │ learning_stats │ dashboard        │
├─────────────────────────────────────────────────┤
│   PostgreSQL │ Redis │ Neo4j │ ChromaDB │ Nginx   │
├─────────────────────────────────────────────────┤
│         AI Services (Spark / DeepSeek)           │
│         Judging Sandbox (Seccomp / Docker)       │
│         Async Task Queue (Dramatiq)              │
└─────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.12+
- PostgreSQL 14+
- Redis 7+
- Neo4j 5.x
- Node.js 16+ (for frontend build)
- Docker & Docker Compose

## Quick Start
https://github.com/proachhh/OnlineJudgeDeploy.git
---

## Project Structure

```
├── account/           # User authentication
├── agents/            # AI multi-agent system
├── aiChat/            # AI chat & dialogue
├── announcement/      # Announcements
├── conf/              # Judge server configuration
├── contest/           # Contest system
├── dashboard/         # Admin data dashboard
├── feedback/          # User feedback
├── judge/             # Judging engine
├── knowledge_graph/   # Knowledge graph
├── learning_stats/    # Learning analytics
├── lesson_plan/       # Lesson plans
├── oj/                # Django project config
├── options/           # System options
├── problem/           # Problem bank
├── recommend/         # Recommendation engine
├── submission/        # Code submission & judging
├── utils/             # Utilities
├── deploy/            # Deployment configs
└── frontend_dist/     # Frontend build output (for deployment)
```

---

## Third-Party Acknowledgements

This project is built upon the following excellent open-source libraries:

### Backend Dependencies

| Library | License | Purpose |
|------|------|------|
| Django 3.2 | BSD | Web framework |
| Django REST Framework | BSD | REST API framework |
| psycopg2 | LGPL | PostgreSQL driver |
| dramatiq | LGPLv3 | Async task queue |
| django-redis | BSD | Redis cache backend |
| Pillow | HPND | Image processing |
| XlsxWriter | BSD | Excel export |
| neo4j-driver | Apache 2.0 | Neo4j graph database driver |
| chromadb | Apache 2.0 | Vector database |
| openai | Apache 2.0 | LLM API helper |
| PyTorch | BSD | Deep learning framework |
| scikit-learn | BSD | Machine learning |
| numpy | BSD | Scientific computing |
| gunicorn | MIT | WSGI server |

### Frontend Dependencies

| Library | License | Purpose |
|------|------|------|
| Vue.js 2.x | MIT | Frontend framework |
| Vuex | MIT | State management |
| Vue Router | MIT | Routing |
| Vue I18n | MIT | Internationalization |
| axios | MIT | HTTP client |
| iView | MIT | UI component library |
| Element UI | MIT | UI component library |
| CodeMirror | MIT | Code editor |
| KaTeX | MIT | Math rendering |
| highlight.js | BSD | Syntax highlighting |
| ECharts | Apache 2.0 | Data visualization |
| moment.js | MIT | Date utilities |

This project is extended from [QingdaoU/OnlineJudge](https://github.com/QingdaoU/OnlineJudge) (MIT License), with substantial enhancements including AI multi-agent system, deep learning recommendation engine, knowledge graph, and more. Original copyright notice below.

---

## License

This project is licensed under the [MIT License](LICENSE).

Extended from [QingdaoU/OnlineJudge](https://github.com/QingdaoU/OnlineJudge) (MIT License). Original code copyright:

```
Copyright (c) 2017-present Qingdao University OnlineJudge Contributors
```

New module code copyright belongs to the authors of this project.
