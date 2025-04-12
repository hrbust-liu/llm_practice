LangChanin 工具实践

[LangChain Tutorials](https://python.langchain.com/docs/tutorials/)

自行注册
```
export OPENAI_API_KEY='' # OpenAi LLM
export TAVILY_API_KEY='' # 提供搜索 Tool
```

工具安装
```
sudo yum install python3-devel openssl-devel  -y
yum install -y tmux
sudo yum install python3.11 -y

python3.11 -m venv myenv

pip3 install openai
pip install -qU langchain-core
pip install sentence-transformers
pip install -U langchain-huggingface
pip install -U langchain_anthropic
pip install -U langgraphRE
````