# GraphMind AI Assistant

<p align="center">
    <img src="https://github.com/michael-hoon/graphreader-agent/blob/main/static/graphmind_logo.jpg" alt="GraphMind Logo" width="300" height="300">
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
5. [References](#references)

---

## Features

- Constructs a knowledge graph from a corpus of documents.
- Provides agentic planning & reasoning using LangGraph, based on the [Graphreader paper](https://arxiv.org/abs/2406.14550). 
- Agent will traverse knowledge graph database and update its "notebook" with relevant information to answer user queries.
- Fully containerized with Docker for on-premises hosting.
- LLM serving via Ollama (currently OpenAI API), with predefined structured output.

---

## GraphMind Agent Architecture

The underlying architecture behind the GraphMind Agentic Framework leverages the [Graphreader paper](https://arxiv.org/abs/2406.14550). The diagrams below show the proposed implementation:

<p align="center">
    <img src="https://github.com/michael-hoon/graphreader-agent/blob/main/static/Architecture.jpg" alt="GraphMind Proposed Architecture">
    <img src="https://github.com/michael-hoon/graphreader-agent/blob/main/static/router.jpg" alt="Semantic Router Agent">
    <img src="https://github.com/michael-hoon/graphreader-agent/blob/main/static/research_subgraph.jpg" alt="Research Subgraph">
    <img src="https://github.com/michael-hoon/graphreader-agent/blob/main/static/agent_nodes.png" alt="Agent Node Architecture">
</p>

The Neo4j Knowledge Graph ontology is pre-defined and shown as follows:


<p align="center">
    <img src="https://github.com/michael-hoon/graphreader-agent/blob/main/static/neo4j_ontology.png" alt="Neo4j Ontology">
</p>

## Requirements

Ensure the following are installed on your system:

- [Python 3.11 or later](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/engine/install/) & Docker Compose (currently not needed)
- [Git](https://git-scm.com/downloads)
- [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) (for Python dependency management [Optional], **or install via pip below**)

---

## Clone the Repository

Start with cloning the repository into your local drive.

```bash
git clone https://github.com/michael-hoon/graphreader-agent.git
cd graphreader-agent
```

## Container Setup

> Note: Current version does not have docker-compose setup yet, have to manually setup containers for now. Requires Neo4j and MinIO S3 images from Docker hub.

### Neo4j Container

Start up another terminal instance. Pull the official Neo4j image from the docker hub repository:

```bash
docker pull neo4j
```

Start an instance of the database:

```bash
docker run \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$HOME/neo4j/data:/data \
    neo4j
```

Visit the WebUI host on http://localhost:7474/browser/, and login with the default username/password: neo4j:neo4j. The UI will prompt you to set a different password. Save this for use in the `.env` file later.

### MinIO Container

Run the following command (**on another instance of your terminal**) to start a container instance of the MinIO S3 object storage (automatically pulls the latest image). Remember to change the installation path of the cloned repository. 

```bash
docker run \
    -p 9000:9000 \
    -p 9001:9001 \
    --name graphreaderminio \
    -v /path-to-your-installation/graphreader-agent/docs:/pdfdata \
    quay.io/minio/minio server /pdfdata --console-address ":9001"
```

Visit the WebUI host on http://127.0.0.1:9001/ and login with the default credentials minioadmin:minioadmin. Change your username/password if required, and remember these details for the `.env` file later. 

#### S3 Bucket Creation

Create a bucket and name it appropriately (e.g. `graphreader-docs`), without any of the features selected. You should see an option to 'Browse Files', select it and upload the two sample PDF files from `/graphreader-agent/docs/` or your own personal PDF files.

#### (Optional) Verify Containers

Verify that all relevant containers are running:

```bash
docker ps
```

---

## Setup Instructions


### 1. Set Up Virtual Environment

On the first terminal instance where you cloned the project repository, create a virtual environment in the `graphreader-agent` directory with `uv` and activate it:

```bash
pip install uv # ignore if uv is already installed on system
uv sync --frozen
# "uv sync" creates .venv automatically

source .venv/bin/activate
```

### 2. Configure the `.env` file

Copy the `.env.example` file and customize it:

```bash
cp .env.example .env
```

Edit the file accordingly. You will require your OpenAI API key, Neo4j database and MinIO S3 storage details from before.

### 3. Running the Application

1. Start the Streamlit frontend:

```bash
cd /src/streamlit_app
streamlit run app.py
```

2. Access the app in your browser at http://localhost:8501.

3. On the sidebar, select the "Process Knowledge Graph" option to run the agent and process your PDF files into the Neo4j database. You can now go to the Neo4j Web console to see the resulting Knowledge Graph.

---

✨ Interact with the GraphMind assistant and analyze your PDF research corpus. Ask any question!

✨ Give feedback on LLM answers and export conversation history on the "Chat Management" tab in JSON format, with human feedback tracked.

## Known Issues / Future Updates
 
1. Currently no functionality for choosing different models on sidebar.
2. Ollama structured output still having issues, using OpenAI API for now.
3. PostgreSQL integration failing due to issue with tracking session states.
4. Agent implementation needs improvement by rewriting certain agents as tools instead.
5. Agent "thoughts" not yet configured properly in streamlit containers, rewriting agents as tools might help. Rerunning streamlit instance removes the container too.
6. FastAPI not integrated yet for Multi-user support on deployed platform
7. Containerisation not configured for graphreader-agent code.
8. Container management with Kubernetes not yet set up.
9. Monitoring/Observability platforms (Grafana) not set up as well.

### References

- [Medium Blog by Neo4j](https://towardsdatascience.com/implementing-graphreader-with-neo4j-and-langgraph-e4c73826a8b7)
- [LangChain documentation](https://python.langchain.com/docs/introduction/)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/tutorials/introduction/)