from openai import OpenAI
client = OpenAI(
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "你是谁"}
  ]
)

print(completion.choices[0].message);
