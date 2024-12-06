import logging

import typer
from dotenv import load_dotenv

from workshop_llm_agents.agents.chatbot_with_tools import graph

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command(
    help="Chatbot with tools",
)
def chatbot_with_tools(
    verbose: bool = False,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    config = {
        "configurable": {
            "thread_id": "1",
        },
    }

    while True:
        exit_code = "q"
        query = input(f"Enter a query(type '{exit_code}' to exit): ")
        if query == exit_code:
            break

        events = graph.stream(
            input={
                "messages": [
                    ("user", query),
                ]
            },
            config=config,
            stream_mode="values",
        )
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()


@app.command(
    help="Export the graph to a PNG file",
)
def export(
    png: str = None,
    verbose: bool = False,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    print(graph.get_graph().draw_mermaid())

    if png:
        graph.get_graph().draw_mermaid_png(
            output_file_path=png,
        )


if __name__ == "__main__":
    load_dotenv()
    app()
