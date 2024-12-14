from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langchain.tools.retriever import create_retriever_tool
from langgraph.managed.is_last_step import RemainingSteps

model_name = "llama3.2"
vectorstore = Chroma(collection_name="netmanuals", persist_directory="./chroma_langchain_db", embedding_function=OllamaEmbeddings(model=model_name))
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_network_manuals",
    "Call this only if required. Search and return information about networking components from their manuals",
)

tools = [retriever_tool]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: RemainingSteps

### Edges

def agent():
    model = ChatOllama(
        model=model_name,
        temperature=0,
        )
    

    def grade_documents(state) -> Literal["generate", "rewrite"]:
        class grade(BaseModel):
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")
        

        llm_with_tool = model.with_structured_output(grade)
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        chain = prompt | llm_with_tool

        messages = state["messages"]
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content

        scored_result = chain.invoke({"question": question, "context": docs})

        score = scored_result.binary_score
        if state["remaining_steps"] <= 20:
            return "generate"

        if score == "yes":
            return "generate"

        return "rewrite"
        
    def agent(state):
        messages = state["messages"]
        model_ = model.bind_tools(tools)
        response = model_.invoke(messages)
        return {"messages": [response]}


    def rewrite(state):
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
            )
        ]
        response = model.invoke(msg)
        return {"messages": [HumanMessage(content=response.content)]}


    def generate(state):
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]

        docs = last_message.content

        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to enhance your answer to the question. 
            Provide the best possible answer, Keep your answers short and concise.
            Question: {question}
            Additional Context: {context}""",
            input_variables=["context", "question"],
        )

        rag_chain = prompt | model | StrOutputParser()

        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}


    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)  
    retrieve = ToolNode([retriever_tool])
    workflow.add_node("retrieve", retrieve)  
    workflow.add_node("rewrite", rewrite)  
    workflow.add_node(
        "generate", generate
    )
    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    return workflow.compile()

if __name__ == "__main__":

    import pprint
    graph = agent()
    inputs = {
        "messages": [
            ("user", "What is the maximum power allowed for 5G?"),
        ]
    }
    for output in graph.stream(inputs):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")
    