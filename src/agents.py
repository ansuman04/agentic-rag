
from crewai import Agent, LLM
from src.tools import DocumentSearchTool
from src.web_tools import WebSearchTool


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


def create_web_search_agent(
    tool: WebSearchTool,
) -> Agent:
    llm = LLM(
        model="qwen2.5:7b",
        custom_openai=True,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0,
    )

    return Agent(
        role="Web Search Specialist",
        goal=(
            "Find relevant and recent information from "
            "public web search results while preserving "
            "the original source URLs."
        ),
        backstory=(
            "You are a web research specialist. "
            "Always use the web_search tool to retrieve "
            "information before responding. "
            "Use only retrieved search results as evidence. "
            "Preserve titles and source URLs. "
            "Never invent sources or URLs. "
            "If search results are insufficient, say so."
        ),
        tools=[tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_web_summary_agent() -> Agent:
    llm = LLM(
        model="qwen2.5:7b",
        custom_openai=True,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0,
    )

    return Agent(
        role="Web Evidence Summarizer",
        goal=(
            "Summarize provided web evidence accurately "
            "without changing source URLs or inventing facts."
        ),
        backstory=(
            "You are a research assistant who summarizes "
            "web evidence already retrieved by another service. "
            "You must use only the supplied evidence. "
            "You cannot perform additional web searches."
        ),
        llm=llm,
        tools=[],
        verbose=True,
        allow_delegation=False,
    )
