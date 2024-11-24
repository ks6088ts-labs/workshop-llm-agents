# How to use

## Scripts

### cosmosdbs.py

```shell
# help
poetry run python scripts/cosmosdbs.py --help

# insert data to Cosmos DB
poetry run python scripts/cosmosdbs.py insert-data \
    --pdf-url "https://www.maff.go.jp/j/wpaper/w_maff/r5/pdf/zentaiban_20.pdf"

# query data from Cosmos DB
poetry run python scripts/cosmosdbs.py query-data \
    --query "農林⽔産祭天皇杯受賞者"
```
