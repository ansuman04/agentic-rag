
from crewai import Agent, LLM
from src.tools import DocumentSearchTool



def create_router_agent() -> Agent:
    llm = LLM(
        model="qwen2.5:7b",
        custom_openai=True,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0,
    )

    return Agent(
        role="Question Router",
        goal=(
            "Decide whether a question should be answered "
            "using the local document knowledge base or web search."
        ),
        backstory=(
            "You are a routing specialist for an Agentic RAG system. "
            "The local knowledge base contains the research paper "
            "'Attention Is All You Need', covering Transformers, "
            "self-attention, positional encoding, and architecture. "
            "Questions requiring current or external information "
            "should use web search."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

def create_retriever_agent(
    tool: DocumentSearchTool,
) -> Agent:
    llm = LLM(
        model="qwen2.5:7b",
        custom_openai=True,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0,
    )

    return Agent(
        role="Document Retriever",
        goal=(
            "Retrieve relevant, verifiable evidence from "
            "the local document knowledge base."
        ),
        backstory=(
            "You are a document retrieval specialist. "
            "Use the document_search tool to retrieve relevant "
            "passages. Preserve source filenames and page numbers. "
            "Do not invent citations or document content."
        ),
        tools=[tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
