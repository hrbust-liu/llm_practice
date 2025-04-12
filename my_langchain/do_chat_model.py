import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

def test_v1():
  from langchain_core.messages import HumanMessage, SystemMessage

  messages = [
      SystemMessage("你是一个小学老师，善于计算"),
      HumanMessage("计算一下 10 * 20!"),
  ]

  model.invoke(messages)
  for token in model.stream(messages):
      # print(token.content, end="|")
      print(token.content, end="")

def test_v2():
  from langchain_core.prompts import ChatPromptTemplate

  system_template = "你是一个小学老师，善于 {math_problem}"

  prompt_template = ChatPromptTemplate.from_messages(
      [("system", system_template), ("user", "{text}")]
  )
  prompt = prompt_template.invoke({"math_problem": "找到数字中的偶数", "text": "1 3 5 6 8 9 13 4314 13"})
  prompt.to_messages()

  response = model.invoke(prompt)
  print(response.content)