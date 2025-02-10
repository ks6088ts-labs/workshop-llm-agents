import logging
from os import getenv

import typer
from azure.identity import (
    ChainedTokenCredential,
    ClientSecretCredential,
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from dotenv import load_dotenv
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

load_dotenv(override=True)
logger = logging.getLogger(__name__)
app = typer.Typer()


def get_azure_ad_token_provider():
    token_credential = ChainedTokenCredential(
        ClientSecretCredential(
            client_id=getenv("AZURE_CLIENT_ID"),
            tenant_id=getenv("AZURE_TENANT_ID"),
            client_secret=getenv("AZURE_CLIENT_SECRET"),
        ),
        DefaultAzureCredential(),
    )
    return get_bearer_token_provider(token_credential, "https://cognitiveservices.azure.com/.default")


def get_azure_openai(service_principal: bool):
    if service_principal:
        return AzureOpenAI(
            azure_ad_token_provider=get_azure_ad_token_provider(),
            api_version=getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT"),
        )
    return AzureOpenAI(
        api_key=getenv("AZURE_OPENAI_API_KEY"),
        api_version=getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT"),
    )


def get_azure_chat_openai(
    service_principal: bool,
    temperature: float = 0,
):
    if service_principal:
        return AzureChatOpenAI(
            temperature=temperature,
            azure_ad_async_token_provider=get_azure_ad_token_provider(),
            api_version=getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT"),
            model=getenv("AZURE_OPENAI_MODEL_GPT"),
        )
    return AzureChatOpenAI(
        temperature=temperature,
        api_key=getenv("AZURE_OPENAI_API_KEY"),
        api_version=getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT"),
        model=getenv("AZURE_OPENAI_MODEL_GPT"),
    )


def get_azure_ai_chat_completion_model(
    service_principal: bool,
    temperature: float = 0,
):
    if service_principal:
        return AzureAIChatCompletionsModel(
            temperature=temperature,
            endpoint=getenv("AZURE_AI_FOUNDRY_INFERENCE_ENDPOINT"),
            credential=DefaultAzureCredential(),
            model_name=getenv("AZURE_OPENAI_MODEL_GPT"),
        )
    return AzureAIChatCompletionsModel(
        temperature=temperature,
        endpoint=getenv("AZURE_AI_FOUNDRY_INFERENCE_ENDPOINT"),
        credential=getenv("AZURE_AI_FOUNDRY_INFERENCE_CREDENTIAL"),
        model_name=getenv("AZURE_OPENAI_MODEL_GPT"),
    )


@app.command()
def openai(
    service_principal: bool = False,
    verbose: bool = False,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    client = get_azure_openai(service_principal)

    chat_completion = client.chat.completions.create(
        model=getenv("AZURE_OPENAI_MODEL_GPT"),
        messages=[
            {
                "role": "user",
                "content": "Hello",
            },
        ],
    )

    print(chat_completion.choices[0].message.content)


@app.command()
def langchain(
    service_principal: bool = False,
    verbose: bool = False,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    llm = get_azure_chat_openai(service_principal)

    response = llm.invoke(
        input="Hello",
    )

    print(response.content)


@app.command()
def langchain_azure_ai_foundry(
    service_principal: bool = False,
    verbose: bool = False,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    llm = get_azure_ai_chat_completion_model(service_principal)

    response = llm.invoke(
        input="Hello",
    )

    print(response.content)


if __name__ == "__main__":
    app()
