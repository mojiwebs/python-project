from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import OpenAIRateLimitError

from dotenv import load_dotenv

load_dotenv()


def main():
    model = ChatOpenAI(temperature=0)
    tools = []
    agent_executor = create_agent(model=model, tools=tools)

    print("Welcome to the LangGraph REACT Agent!")
    print("ask me to calculate anything")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit" or user_input.lower() == "quit":
            print("Exiting the agent. Goodbye!")
            break
        print("\nAgent: ", end="")
        try:
            for state in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]},
                stream_mode="values",
            ):
                message = state["messages"][-1]
                if message.type == "ai" and message.content:
                    print(message.content, end="")
        except OpenAIRateLimitError:
            print(
                "OpenAI rejected the request because this API account has no "
                "credits remaining. Add billing credits or use another model provider."
            )
        print()  # Print a newline after streaming output


if __name__ == "__main__":
    main()
