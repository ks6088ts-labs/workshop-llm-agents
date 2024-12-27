# Agents

## Overview

### Architecture

![Agents](./images/workshop-llm-agents.png)

## How to run

### Local

Before running codes, you need to install the dependencies.

```shell
$ poetry install
```

To see some components are properly working, you can see the available commands as below.

```shell
$ poetry run python main.py --help
```

To run the Streamlit app, you can run the following commands.

```shell
$ poetry run streamlit run main.py streamlit-app
```

### Docker

```shell
# see the available commands
$ docker run --rm \
    -v ${PWD}/.env:/app/.env \
    ks6088ts/workshop-llm-agents:latest \
    python main.py --help

# mount the .env file to the container and expose the port 8501
$ docker run --rm \
    -v ${PWD}/.env:/app/.env \
    -p 8501:8501 \
    ks6088ts/workshop-llm-agents:latest
```

## Use cases

### Chatbot with Tools

Graph:

![Chatbot with Tools](./images/chatbot_with_tools.png)

Screenshot:

![Chatbot with Tools Web Interface](./images/1_chatbot_with_tools.png)

#### References

- [🚀 LangGraph Quick Start > Part 2: 🛠️ Enhancing the Chatbot with Tools](https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-2-enhancing-the-chatbot-with-tools)

### [Research Agent with Human-in-the-loop](https://github.com/mahm/softwaredesign-llm-application/tree/main/14)

Graph:

![Research Agent with Human-in-the-loop](./images/research_agent.png)

Screenshot:

![Research Agent with Human-in-the-loop](./images/2_research_agent_with_human_in_the_loop.png)

YouTube: [Research Agent using Azure](https://youtu.be/7Tp_TvTpuw8)

#### References

- [Research Agent with Human-in-the-loop](https://github.com/mahm/softwaredesign-llm-application/tree/main/14)

### Documentation Agent

Graph:

![Documentation Agent](./images/documentation_agent.png)

Screenshot:

![Documentation Agent](./images/3_documentation_agent.png)

### Tools Agent

Graph:

![Tools Agent](./images/tools_agent.png)

Screenshot:

![Tools Agent](./images/4_tools_agent.png)

### Summarize Agent

Graph:

![Summarize Agent](./images/summarize_agent.png)

```shell
$ poetry run python main.py agents-summarize-run \
    --png ./docs/images/summarize_agent.png
```

#### References

- [How to summarize text through parallelization](https://python.langchain.com/docs/how_to/summarize_map_reduce/)
