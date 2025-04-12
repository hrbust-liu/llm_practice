import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

def example():
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_openai import ChatOpenAI
  from pydantic import BaseModel, Field

  tagging_prompt = ChatPromptTemplate.from_template(
      """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}
    """
  )

  class Classification(BaseModel):
      sentiment: str = Field(description="The sentiment of the text")
      aggressiveness: int = Field(
          description="How aggressive the text is on a scale from 1 to 10"
      )
      language: str = Field(description="The language the text is written in")


  # LLM
  llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
      Classification
  )

  inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
  prompt = tagging_prompt.invoke({"input": inp})
  response = llm.invoke(prompt)

  print(response)

def my_test():
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_openai import ChatOpenAI
  from pydantic import BaseModel, Field

  tagging_prompt = ChatPromptTemplate.from_template(
      """
  你是一个计算机工程师，善于分析程序, 从以下段落中提取所需信息。

仅提取“Classification”函数中提到的属性。

段落：
{input}

  """
  )

  class Classification(BaseModel):
      has_error: bool = Field(description="代码是否错误")
      num: int = Field(
          description="代码变量有多少个"
      )
      language: str = Field(description="代码是什么语言")


  # LLM
  llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
      Classification
  )

  inp = " int a, b, c = 0; d++, b++;"
  prompt = tagging_prompt.invoke({"input": inp})
  response = llm.invoke(prompt)

  print(response)

my_test()