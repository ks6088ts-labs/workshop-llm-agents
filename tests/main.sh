#!/bin/bash

set -eux

# Run tests for main.py commands
commands=(
  # llms
  llms-azure-openai-chat
  llms-azure-openai-embeddings
  # tasks
  tasks-passive-goal-creator
  tasks-prompt-optimizer
  tasks-query-decomposer
  tasks-task-executor
  # tools
  tools-bing-search
  # vector-stores
  # vector-stores-cosmosdb-insert-data
  vector-stores-cosmosdb-query-data
)

# See all available commands
poetry run python main.py --help

for command in "${commands[@]}"; do
  poetry run python main.py "$command" --verbose
done

# Run Streamlit app
# poetry run streamlit run main.py streamlit-app
