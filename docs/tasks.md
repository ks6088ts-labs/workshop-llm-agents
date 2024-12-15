# Tasks

## ScoringEvaluator

```shell
$ poetry run python main.py tasks-scoring-evaluator \
    --query "something wrong with my computer"

# score=1 reason='The bug report lacks essential details. It does not specify the model number of the hardware or the version of the software. There are no reproduction steps provided, nor is there any information on the scope or impact of the bug. The urgency of the issue is also not mentioned. Overall, the report is too vague to be actionable.'
```

### References

- [Scoring Evaluator](https://python.langchain.com/v0.1/docs/guides/productionization/evaluation/string/scoring_eval_chain/)
- [libs/langchain/langchain/evaluation/scoring](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/evaluation/scoring)

## ImageLabeler

```shell
$ poetry run python main.py tasks-image-labeler \
    --file ./docs/images/workshop-llm-agents.png

# Labels: [<Label.AZURE: 'Azure'>, <Label.LANGCHAIN: 'Langchain'>]
```

### References

- []()
