from typing import Optional

from pydantic import BaseModel, Field
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )
# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)
def example():
    structured_llm = llm.with_structured_output(schema=Person)
    text = "Alan Smith is 6 feet tall and has blond hair."
    prompt = prompt_template.invoke({"text": text})
    response = structured_llm.invoke(prompt)
    print(response)

from typing import List, Optional

class MyAnimal(BaseModel):
    """Information about a person."""
    animal_num: Optional[str] = Field(default=None, description="我家有多少个宠物`")
    animal_type: Optional[str] = Field(default=None, description="宠物都是什么动物`")
    animal_name: Optional[str] = Field(default=None, description="宠物名字都是什么`")

class MyFamily(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[MyAnimal]


def my_test():
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            # Please see the how-to about improving performance with
            # reference examples.
            # MessagesPlaceholder('examples'),
            ("human", "{text}"),
        ]
    )

    structured_llm = llm.with_structured_output(schema=MyFamily)
    text = "我家有俩只狗，一只狗叫旺财, 另一只狗旺旺, 还有只猫叫招财."
    prompt = prompt_template.invoke({"text": text})
    response = structured_llm.invoke(prompt)
    print(response)
my_test()