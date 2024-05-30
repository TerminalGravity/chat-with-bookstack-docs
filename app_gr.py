import os
from typing import Optional, Tuple, List

import gradio as gr
from threading import Lock
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.manager import trace_as_chain_group

# Load environment variables from .env file
load_dotenv()

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    # Create embeddings
    embeddings = CohereEmbeddings()

    # Create vector stores and retrievers
    vectorstore1 = Chroma(embedding_function=embeddings, persist_directory="chroma1")
    retriever1 = vectorstore1.as_retriever()
    vectorstore2 = Chroma(embedding_function=embeddings, persist_directory="chroma2")
    retriever2 = vectorstore2.as_retriever()
    vectorstore3 = Chroma(embedding_function=embeddings, persist_directory="chroma3")
    retriever3 = vectorstore3.as_retriever()
    vectorstore4 = Chroma(embedding_function=embeddings, persist_directory="chroma4")
    retriever4 = vectorstore4.as_retriever()

    retrievers = {
        "Regence BCBS SPC FEP Program": retriever1,
        "Regence BCBS UT FEP Program": retriever2,
        "Kynetec": retriever3,
        "ShareCare Program": retriever4
    }

    # Define document prompt and LLM chain
    document_prompt = PromptTemplate(input_variables=["page_content"], template="{page_content}")
    document_variable_name = "context"
    llm = ChatOpenAI(temperature=0.2)
    system_prompt = SystemMessagePromptTemplate.from_template("""Use the following pieces of context to answer user questions. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n--------------\n\n{context}""")
    prompt = ChatPromptTemplate(
        messages=[
            system_prompt,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    combine_docs_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
        document_separator="---------"
    )

    # Define question generator chain
    question_template = """Combine the chat history and follow up question into a search query.\n\nChat History:\n\n{chat_history}\n\nFollow up question: {question}"""
    question_prompt = PromptTemplate.from_template(question_template)
    llm = ChatOpenAI(temperature=0)
    question_generator_chain = LLMChain(llm=llm, prompt=question_prompt)

    return combine_docs_chain, question_generator_chain, retrievers

class ChatWrapper:
    def __init__(self, agent_state):
        self.lock = Lock()
        self.agent_state = agent_state

    def __call__(self, inp: str, history: Optional[List[Tuple[str, str]]], selected_program: str):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            combine_docs_chain, question_generator_chain, retrievers = self.agent_state
            current_retriever = retrievers[selected_program]

            # Set OpenAI key
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")

            convo_string = "\n\n".join([f"Human: {h}\nAssistant: {a}" for h, a in history])
            messages = []
            for human, ai in history:
                messages.append(HumanMessage(content=human))
                if isinstance(ai, str):
                    messages.append(AIMessage(content=ai))
                else:
                    ai = ai.get("content", "")
                    messages.append(AIMessage(content=ai))

            with trace_as_chain_group("qa_response") as group_manager:
                search_query = question_generator_chain.invoke(
                    input={
                        "question": inp,
                        "chat_history": convo_string
                    },
                    callbacks=group_manager
                )

                docs = current_retriever.get_relevant_documents(
                    query=search_query["text"],
                    callbacks=group_manager
                )

                response = combine_docs_chain.invoke(
                    input={
                        "input_documents": docs,
                        "chat_history": messages,
                        "question": inp
                    },
                    callbacks=group_manager
                )

            if isinstance(response, dict):
                final_response = response.get("output_text", "No response generated.")
            else:
                final_response = response

            updated_history = history + [(inp, final_response)]

        except Exception as e:
            raise e
        finally:
            self.lock.release()

        return updated_history, updated_history

# Initialize agent state
agent_state = load_chain()
chat = ChatWrapper(agent_state)

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>LangChain Demo</center></h3>")

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask a question related to the Regence BCBS or Kynetec programs",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary")

    gr.Examples(
        examples=[
            "What are the steps for generating the Card Fulfillment Activity report for Regence BCBS SPC?",
            "How do I manage weekly check orders for Kynetec?",
            "Can you explain the invoicing process for the ShareCare Program?",
            "What are the key tasks for the Navigate Wellbeing Solutions rewards program?"
        ],
        inputs=message,
    )

    gr.HTML("Demo application of a LangChain chain.")

    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )

    state = gr.State()

    radio = gr.Radio(
        choices=["Regence BCBS SPC FEP Program", "Regence BCBS UT FEP Program", "Kynetec", "ShareCare Program"],
        label="Select Client Program",
        value="Regence BCBS SPC FEP Program"
    )

    submit.click(chat, inputs=[message, state, radio], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state, radio], outputs=[chatbot, state])

block.launch(debug=True, share=True)


import os
from typing import Optional, Tuple, List

import gradio as gr
from threading import Lock
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.manager import trace_as_chain_group

# Load environment variables from .env file
load_dotenv()

def load_chain(temperature: float):
    """Logic for loading the chain you want to use should go here."""
    # Create embeddings
    embeddings = CohereEmbeddings()

    # Create vector stores and retrievers
    vectorstore1 = Chroma(embedding_function=embeddings, persist_directory="chroma1")
    retriever1 = vectorstore1.as_retriever()
    vectorstore2 = Chroma(embedding_function=embeddings, persist_directory="chroma2")
    retriever2 = vectorstore2.as_retriever()
    vectorstore3 = Chroma(embedding_function=embeddings, persist_directory="chroma3")
    retriever3 = vectorstore3.as_retriever()
    vectorstore4 = Chroma(embedding_function=embeddings, persist_directory="chroma4")
    retriever4 = vectorstore4.as_retriever()

    retrievers = {
        "Regence BCBS SPC FEP Program": retriever1,
        "Regence BCBS UT FEP Program": retriever2,
        "Kynetec": retriever3,
        "ShareCare Program": retriever4
    }

    # Define document prompt and LLM chain
    document_prompt = PromptTemplate(input_variables=["page_content"], template="{page_content}")
    document_variable_name = "context"
    llm = ChatOpenAI(temperature=temperature)
    system_prompt = SystemMessagePromptTemplate.from_template("""Use the following pieces of context to answer user questions. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n--------------\n\n{context}""")
    prompt = ChatPromptTemplate(
        messages=[
            system_prompt,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    combine_docs_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
        document_separator="---------"
    )

    # Define question generator chain
    question_template = """Combine the chat history and follow up question into a search query.\n\nChat History:\n\n{chat_history}\n\nFollow up question: {question}"""
    question_prompt = PromptTemplate.from_template(question_template)
    llm = ChatOpenAI(temperature=0)
    question_generator_chain = LLMChain(llm=llm, prompt=question_prompt)

    return combine_docs_chain, question_generator_chain, retrievers

from threading import Lock
import os
from typing import List, Optional, Tuple
import gradio as gr
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.manager import trace_as_chain_group

class ChatWrapper:
    def __init__(self, agent_state):
        self.lock = Lock()
        self.agent_state = agent_state

    def __call__(self, inp: str, history: Optional[List[Tuple[str, str]]], selected_program: str, temperature: float):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            combine_docs_chain, question_generator_chain, retrievers = self.agent_state
            current_retriever = retrievers[selected_program]

            # Set OpenAI key
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")

            convo_string = "\n\n".join([f"Human: {h}\nAssistant: {a}" for h, a in history])
            messages = []
            for human, ai in history:
                messages.append(HumanMessage(content=human))
                if isinstance(ai, str):
                    messages.append(AIMessage(content=ai))
                else:
                    ai = ai.get("content", "")
                    messages.append(AIMessage(content=ai))

            with trace_as_chain_group("qa_response") as group_manager:
                search_query = question_generator_chain.invoke(
                    input={
                        "question": inp,
                        "chat_history": convo_string
                    },
                    callbacks=group_manager
                )

                docs = current_retriever.get_relevant_documents(
                    query=search_query["text"],
                    callbacks=group_manager
                )

                response = combine_docs_chain.invoke(
                    input={
                        "input_documents": docs,
                        "chat_history": messages,
                        "question": inp
                    },
                    callbacks=group_manager
                )

            if isinstance(response, dict):
                final_response = response.get("output_text", "No response generated.")
            else:
                final_response = response

            updated_history = history + [(inp, final_response)]

        except Exception as e:
            raise e
        finally:
            self.lock.release()

        return updated_history, updated_history

# Initialize agent state
agent_state = load_chain(temperature=0.2)  # Default temperature
chat = ChatWrapper(agent_state)

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>LangChain Demo</center></h3>")

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask a question related to the Regence BCBS or Kynetec programs",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary")

    gr.Examples(
        examples=[
            "What are the steps for generating the Card Fulfillment Activity report for Regence BCBS SPC?",
            "How do I manage weekly check orders for Kynetec?",
            "Can you explain the invoicing process for the ShareCare Program?",
            "What are the key tasks for the Navigate Wellbeing Solutions rewards program?"
        ],
        inputs=message,
    )


    state = gr.State()

    radio = gr.Radio(
        choices=["Regence BCBS SPC FEP Program", "Regence BCBS UT FEP Program", "Kynetec", "ShareCare Program"],
        label="Select Client Program",
        value="Regence BCBS SPC FEP Program"
    )

    temperature_slider = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=0.2,
        step=0.1,
        label="Temperature"
    )

    def update_agent_state(temperature):
        global agent_state
        agent_state = load_chain(temperature)
        chat.agent_state = agent_state

    temperature_slider.change(update_agent_state, inputs=temperature_slider, outputs=[])

    submit.click(chat, inputs=[message, state, radio, temperature_slider], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state, radio, temperature_slider], outputs=[chatbot, state])

block.launch(debug=True, share=True)