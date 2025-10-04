from itertools import chain
import os
from dotenv import load_dotenv
load_dotenv()
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatOpenAI(temperature=0, model="gpt-4")
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=llm, 
    tools=tools, 
    prompt=react_prompt
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

def main():
    print("Hello from langchain-course!")
    result = agent_executor.invoke(
        input={
            "input": "Search for 3 job posting for AI engineer using Langchain in Nigeria from LinkedIn and list their details",
            }
    )
    print(result)


if __name__ == "__main__":
    main()
