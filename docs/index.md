# How to use

## cosmosdbs.py

To use Microsoft Entra ID authentication, you need to create a role assignment for the user or service principal.
Refer to the following documents for more information.

- [Use data plane role-based access control with Azure Cosmos DB for NoSQL](https://learn.microsoft.com/azure/cosmos-db/how-to-setup-rbac#role-assignments)
- [Use data plane role-based access control with Azure Cosmos DB for NoSQL](https://learn.microsoft.com/azure/cosmos-db/nosql/security/how-to-grant-data-plane-role-based-access?tabs=built-in-definition%2Cpython&pivots=azure-interface-cli)

To create a role assignment for the user or service principal, you can use the following Azure CLI commands.

```shell
# Set variables
RESOURCE_GROUP_NAME="YOUR_RESOURCE_GROUP_NAME"
COSMOSDB_ACCOUNT_NAME="YOUR_COSMOSDB_ACCOUNT_NAME"
# Note: If you are creating a role assignment for a service principal, use the Object ID in the Enterprise applications section of the Microsoft Entra ID portal blade.
PRINCIPAL_ID="00000000-0000-0000-0000-000000000000"

# Get the role definition ID
# ROLE_NAME="Cosmos DB Built-in Data Reader"
ROLE_NAME="Cosmos DB Built-in Data Contributor"
ROLE_DEFINITION_ID=$(az cosmosdb sql role definition list \
    --resource-group $RESOURCE_GROUP_NAME \
    --account-name $COSMOSDB_ACCOUNT_NAME \
    --query "[?roleName=='$ROLE_NAME'].id" --output tsv)

# Get the Cosmos DB account ID
AZURE_COSMOSDB_ACCOUNT_ID=$(az cosmosdb show \
    --resource-group $RESOURCE_GROUP_NAME \
    --name $COSMOSDB_ACCOUNT_NAME \
    --query "{id:id}" --output tsv)

# Assign the role to the user
az cosmosdb sql role assignment create \
    --resource-group $RESOURCE_GROUP_NAME \
    --account-name $COSMOSDB_ACCOUNT_NAME \
    --role-definition-id $ROLE_DEFINITION_ID \
    --scope $AZURE_COSMOSDB_ACCOUNT_ID \
    --principal-id $PRINCIPAL_ID

# List the role assignments
az cosmosdb sql role assignment list \
    --resource-group $RESOURCE_GROUP_NAME \
    --account-name $COSMOSDB_ACCOUNT_NAME
```

Run the following commands to use the `cosmosdbs.py` script.

```shell
# help
poetry run python scripts/cosmosdbs.py --help

# insert data to Cosmos DB
poetry run python scripts/cosmosdbs.py insert-data \
    --pdf-url "https://www.maff.go.jp/j/wpaper/w_maff/r5/pdf/zentaiban_20.pdf" \
    --verbose --service-principal

# query data from Cosmos DB
poetry run python scripts/cosmosdbs.py query-data \
    --query "農林⽔産祭天皇杯受賞者" \
    --verbose --service-principal
```

### References

- [Azure Cosmos DB No SQL](https://python.langchain.com/docs/integrations/vectorstores/azure_cosmos_db_no_sql/)
- [Learn Azure Azure Cosmos DB Vector database](https://learn.microsoft.com/azure/cosmos-db/vector-database)
- [AzureDataRetrievalAugmentedGenerationSamples/Python/CosmosDB-NoSQL_VectorSearch](https://github.com/microsoft/AzureDataRetrievalAugmentedGenerationSamples/tree/main/Python/CosmosDB-NoSQL_VectorSearch)
- [Azure Cosmos DB ベクター検索機能と RAG の実装ガイド](https://note.com/generativeai_new/n/n3fcb2e57d195)
- [Azure CosmosDB for NoSQL でベクトル検索しよう！！](https://zenn.dev/nomhiro/articles/cosmos-nosql-vector-search)
- [Use data plane role-based access control with Azure Cosmos DB for NoSQL](https://learn.microsoft.com/azure/cosmos-db/nosql/security/how-to-grant-data-plane-role-based-access?tabs=built-in-definition%2Ccsharp&pivots=azure-interface-cli)

## bing_searches.py

```shell
# help
poetry run python scripts/bing_searches.py --help

# search data from Bing
poetry run python scripts/bing_searches.py search \
    --query "Who is the CEO of Microsoft?"

# search data from Bing
poetry run python scripts/bing_searches.py chain \
    --query "Who is the CEO of Microsoft?"
```

### References

- [Bing Search](https://python.langchain.com/docs/integrations/tools/bing_search/)

## langgraphs.py

![langgraphs_mermaid](images/langgraphs_mermaid.png)

```shell
# help
poetry run python scripts/langgraphs.py --help

# draw a graph in mermaid format
poetry run python scripts/langgraphs.py draw-mermaid-png \
    --output docs/images/langgraphs_mermaid.png

# run a workflow implemented by LangGraph
poetry run python scripts/langgraphs.py run \
    --query "How is the weather today in Japan?"
poetry run python scripts/langgraphs.py run \
    --query "How is the weather today in San Francisco?"
```

### References

- [🦜🕸️LangGraph](https://langchain-ai.github.io/langgraph/)
- [🚀 LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

## langchains.py

```shell
# help
poetry run python scripts/langchains.py --help

# via OpenAI SDK
# call with API key
poetry run python scripts/langchains.py openai --verbose
# call with service principal
poetry run python scripts/langchains.py openai --verbose --service-principal

# via LangChain
# call with API key
poetry run python scripts/langchains.py langchain --verbose
# call with service principal
poetry run python scripts/langchains.py langchain --verbose --service-principal
```

### References

- [How to switch between OpenAI and Azure OpenAI endpoints with Python > Microsoft Entra ID authentication](https://learn.microsoft.com/azure/ai-services/openai/how-to/switching-endpoints#microsoft-entra-id-authentication)
