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
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

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
        AzureChatOpenAI(
            temperature=temperature,
            azure_ad_async_token_provider=get_azure_ad_token_provider(),
            api_version=getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT"),
            model=getenv("AZURE_OPENAI_GPT_MODEL"),
        )
    return AzureChatOpenAI(
        temperature=temperature,
        api_key=getenv("AZURE_OPENAI_API_KEY"),
        api_version=getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT"),
        model=getenv("AZURE_OPENAI_GPT_MODEL"),
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
        model=getenv("AZURE_OPENAI_GPT_MODEL"),
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


if __name__ == "__main__":
    load_dotenv()
    app()
