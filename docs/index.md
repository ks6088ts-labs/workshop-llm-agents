# Agents

## Overview

### Architecture

![Agents](./images/workshop-llm-agents.png)

## Use cases

### Chatbot with Tools

![Chatbot with Tools](./images/chatbot_with_tools.png)

To implement a chatbot with tools, you can refer to [🚀 LangGraph Quick Start > Part 2: 🛠️ Enhancing the Chatbot with Tools](https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-2-enhancing-the-chatbot-with-tools).

#### CLI

To run the chatbot with tools in the terminal, you can use the following command:

```shell
# help
$ poetry run python main.py --help

# run chatbot with tools in terminal
$ poetry run python main.py chatbot-with-tools

# Enter a query(type 'q' to exit): Please explain LangChain with Web search
# ================================ Human Message =================================

# Please explain LangChain with Web search
# ================================== Ai Message ==================================
# Tool Calls:
#   bing_search_results_json (call_ypoIsXUcpnGoMj4Xa7pvDHaG)
#  Call ID: call_ypoIsXUcpnGoMj4Xa7pvDHaG
#   Args:
#     query: LangChain
# ================================= Tool Message =================================
# Name: bing_search_results_json

# [{'snippet': 'Build your app with <b>LangChain</b> Build context-aware, reasoning applications with <b>LangChain</b>’s flexible framework that leverages your company’s data and APIs. Future-proof your application by making vendor optionality part of your LLM infrastructure design.', 'title': 'LangChain', 'link': 'https://www.langchain.com/'}, {'snippet': '<b>Chainsは</b>、<b>LangChainという</b>ソフトウェア名にもなっているように中心的な機能です。 その名の通り、LangChainが持つ様々な機能を「連結」して組み合わせることができます。 試しに chains.py というファイルを作って以下のコードを書いてみ', 'title': 'そろそろ知っておかないとヤバい？ 話題のLangChainを30分だけ ...', 'link': 'https://qiita.com/minorun365/items/081fc560e08f0197a7a8'}, {'snippet': 'LangChainは、プロンプトエンジニアリングを可能にするPythonやJavaScript・TypeScriptのライブラリです。このページでは、LangChainのインストール方法や主な機能、使い方を紹介します。', 'title': 'LangChainの概要と使い方｜サクッと始めるプロンプト ... - Zenn', 'link': 'https://zenn.dev/umi_mori/books/prompt-engineer/viewer/langchain_overview'}, {'snippet': '<b>LangChain</b> <b>とは何か</b>、<b>企業が</b> LangChain を使用する方法と理由、および AWS で LangChain を使用する方法。 メインコンテンツに移動 アマゾン ウェブ サービスのホームページに戻るには、ここをクリック', 'title': 'LangChain とは何ですか? - LangChain の説明 - AWS', 'link': 'https://aws.amazon.com/jp/what-is/langchain/'}]
# ================================== Ai Message ==================================

# LangChain is a flexible framework designed to build context-aware and reasoning applications by leveraging a company's data and APIs. It allows for the creation of applications that can adapt and make decisions based on the provided context. Here are some key points about LangChain:

# 1. **Core Functionality**: LangChain's core functionality revolves around "chains," which allow for the combination and linking of various features and capabilities within the framework. This modular approach enables developers to create complex applications by connecting different components.

# 2. **Prompt Engineering**: LangChain supports prompt engineering, making it possible to design and optimize prompts for language models. This is particularly useful for applications that require precise and contextually relevant responses.

# 3. **Programming Languages**: LangChain provides libraries for Python, JavaScript, and TypeScript, making it accessible to developers with different programming backgrounds.

# 4. **Vendor Optionality**: One of the key design principles of LangChain is to future-proof applications by incorporating vendor optionality into the LLM (Large Language Model) infrastructure. This means that applications built with LangChain can be more adaptable to changes in underlying technologies and vendors.

# For more detailed information, you can visit the [LangChain website](https://www.langchain.com/).
# Enter a query(type 'q' to exit): q

# Export the chatbot with tools to a PNG file
$ poetry run python main.py export --png docs/images/chatbot_with_tools.png
```

#### Web

To run the chatbot with tools in the web interface, you can use the following command:

```shell
$ poetry run python -m streamlit run workshop_llm_agents/streamlits/1_chatbot_with_tools.py
```

![Chatbot with Tools Web Interface](./images/1_chatbot_with_tools.png)
