# GraphMind AI Assistant

<p align="center">
    <img src="https://github.com/michael-hoon/graphreader-agent/blob/main/static/graphmind_logo.jpg" alt="GraphMind Logo" width="200" height="200">
</p>

GraphMind is an advanced AI research assistant leveraging LangGraph for agent orchestration and Neo4j to construct and query a knowledge graph of research papers. This repository contains the codebase and setup instructions for deploying GraphMind locally.

---

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Setup Instructions](#setup-instructions)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Set Up the Virtual Environment](#2-set-up-the-virtual-environment)
    - [3. Configure the `.env` File](#3-configure-the-env-file)
    - [4. Install Dependencies](#4-install-dependencies)
    - [5. Set Up Docker Containers](#5-set-up-docker-containers)
4. [Running the Application](#running-the-application)
5. [Contributing](#contributing)
6. [License](#license)

---

## Features

- Constructs a knowledge graph from a corpus of documents.
- Provides agentic planning & reasoning using LangGraph, based on the [Graphreader paper](https://arxiv.org/abs/2406.14550). 
- Agent will traverse knowledge graph database and update its "notebook" with relevant information to answer user queries.
- Fully containerized with Docker for on-premises hosting.
- LLM serving via Ollama, with structured output.

---

## GraphMind Agent Architecture

The underlying architecture behind the GraphMind Agentic Framework leverages the [Graphreader paper](https://arxiv.org/abs/2406.14550). The diagrams below show the proposed implementation:

<p align="center">
    <img src="https://github.com/michael-hoon/graphreader-agent/blob/main/static/Architecture.jpg" alt="GraphMind Proposed Architecture" width="200" height="200">
    <img src="https://github.com/michael-hoon/graphreader-agent/blob/main/static/router.jpg" alt="Semantic Router Agent" width="200" height="200">
    <img src="https://github.com/michael-hoon/graphreader-agent/blob/main/static/research_subgraph.jpg" alt="Research Subgraph" width="200" height="200">
</p>

## Requirements

Ensure the following are installed on your system:

- Python 3.8 or later
- Docker & Docker Compose
- Git
- `Poetry` (for dependency management)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/GraphMind.git
cd GraphMind
```

### 2. Set Up Virtual Environment

Create a virtual environment in Python and activate it:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: venv\Scripts\activate
```

### 3. Configure the `.env` file

Copy the `.env.example` file and customize it:

```bash
cp .env.example .env
```

Edit the file accordingly.

### 4. Install Dependencies

Install dependencies using `Poetry`:

```bash
poetry install
```

Alternatively, install directly from the requirements.txt file:

```bash
pip install -r requirements.txt
```

### 5. Set Up Docker Containers

- Ensure Docker is installed and running.
- Build and start the required containers:

```bash
docker-compose up --build -d
```

This will set up:

- **Neo4j**: Graph database for storing the knowledge graph.
- **Ollama**: Local LLM model container.
- **MinIO**: Local S3 object storage for PDF files.
- **PostgreSQL**: Relational Database for miscellaneous user and session information.
- Any additional services defined in the docker-compose.yml.- 

Verify the containers are running:

```bash
docker ps
```

### Running the Application

1. Start the Streamlit frontend:

```bash
cd /src/streamlit_app
streamlit run app.py
```

2. Access the app in your browser at http://localhost:8501.

3. Interact with the GraphMind assistant and analyze your research corpus. Ask any question!

### References

- [Medium Blog by Neo4j](https://towardsdatascience.com/implementing-graphreader-with-neo4j-and-langgraph-e4c73826a8b7)
- [LangChain documentation](https://python.langchain.com/docs/introduction/)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/tutorials/introduction/)