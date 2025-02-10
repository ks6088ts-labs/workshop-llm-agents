# References

## [Open WebUI](https://github.com/open-webui/open-webui)

- [Azure OpenAI Service を使って遊ぶ](https://zenn.dev/watarukura/articles/20240618-yei4zhgmsn39f3mklflu63tenvqejn)
- [docker-compose.yaml](https://github.com/open-webui/open-webui/blob/main/docker-compose.yaml)
- [LiteLLM OpenAI 互換プロキシで異なる LLM に対して同じコードで Function Calling（OpenAI/Anthropic）](https://zenn.dev/kun432/scraps/e1ff3ebfb97177)
- [Open WebUI (Formerly Ollama WebUI) がすごい](https://qiita.com/moritalous/items/1cad6878ea750d18747c)

## [Dify](https://github.com/langgenius/dify)

- [Quick start](https://github.com/langgenius/dify?tab=readme-ov-file#quick-start)

```shell
git clone https://github.com/langgenius/dify.git

cd dify/docker

cp .env.example .env
docker compose up -d

# Reset the password of the admin account
# ref. https://docs.dify.ai/getting-started/install-self-hosted/faqs#id-4.-how-to-reset-the-password-of-the-admin-account
docker exec -it docker-api-1 flask reset-password
```

### Ollama

```shell
# Run the Ollama server locally
export OLLAMA_HOST=http://localhost:11434
ollama serve

# Specify the base url for Ollama: `http://host.docker.internal:11434`
# ref. https://qiita.com/Tadataka_Takahashi/items/ba832511bd4fd5cd46f1
```
