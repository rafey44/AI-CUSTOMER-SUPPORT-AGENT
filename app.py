import os
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Customer Support Agent",
    page_icon="🤖",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("🤖 AI Customer Support Agent")
st.write(
    "Ask questions about orders, customers, products, "
    "shipping, returns, and store policies."
)


# =========================================================
# DATABASE SETUP
# =========================================================

DB_PATH = "customer_support.db"


def create_database():

    customers = pd.read_excel("data/customers.xlsx")
    orders = pd.read_excel("data/orders.xlsx")
    products = pd.read_excel("data/products.xlsx")

    with sqlite3.connect(DB_PATH) as connection:

        customers.to_sql(
            "customers",
            connection,
            if_exists="replace",
            index=False
        )

        orders.to_sql(
            "orders",
            connection,
            if_exists="replace",
            index=False
        )

        products.to_sql(
            "products",
            connection,
            if_exists="replace",
            index=False
        )


create_database()


# =========================================================
# DATABASE FUNCTIONS
# =========================================================

def check_customer(customer_id):

    with sqlite3.connect(DB_PATH) as connection:

        query = """
        SELECT *
        FROM customers
        WHERE customer_id = ?
        """

        result = pd.read_sql_query(
            query,
            connection,
            params=(customer_id,)
        )

    if result.empty:
        return "Customer not found."

    return result


def check_order(order_id):

    with sqlite3.connect(DB_PATH) as connection:

        query = """
        SELECT *
        FROM orders
        WHERE order_id = ?
        """

        result = pd.read_sql_query(
            query,
            connection,
            params=(order_id,)
        )

    if result.empty:
        return "Order not found."

    return result


def get_product(product_id):

    with sqlite3.connect(DB_PATH) as connection:

        query = """
        SELECT *
        FROM products
        WHERE product_id = ?
        """

        result = pd.read_sql_query(
            query,
            connection,
            params=(product_id,)
        )

    if result.empty:
        return "Product not found."

    return result


# =========================================================
# RAG KNOWLEDGE BASE
# =========================================================

# =========================================================
# RAG KNOWLEDGE BASE
# =========================================================

@st.cache_resource
def load_rag():

    knowledge_folder = "knowledge_base"

    chunks = []

    for filename in os.listdir(knowledge_folder):

        if filename.endswith(".txt"):

            file_path = os.path.join(
                knowledge_folder,
                filename
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                text = file.read()

            paragraphs = [
                paragraph.strip()
                for paragraph in text.split("\n\n")
                if paragraph.strip()
            ]

            chunks.extend(paragraphs)

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    ).astype("float32")

    return chunks, embedding_model, embeddings


chunks, embedding_model, embeddings = load_rag()


def search_knowledge_base(question, top_k=3):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    scores = np.dot(
        embeddings,
        query_embedding[0]
    )

    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []

    for i in top_indices:

        if i < len(chunks):
            results.append(chunks[i])

    if not results:
        return "No relevant information found."

    return "\n\n".join(results)
@tool("search_knowledge_base")
def knowledge_base_tool(question: str) -> str:
    """Search store policies and knowledge base."""

    return search_knowledge_base(
        question,
        top_k=3
    )


# =========================================================
# GROQ / CREWAI LLM
# =========================================================

groq_api_key = st.secrets["GROQ_API_KEY"]

llm = LLM(
    model="groq/openai/gpt-oss-20b",
    api_key=groq_api_key
)


# =========================================================
# AGENT
# =========================================================

customer_support_agent = Agent(

    role="Customer Support Agent",

    goal="""
    Help customers using only verified information
    from the store database and knowledge base.
    """,

    backstory="""
    You are a professional customer support agent
    for a fictional technology store.

    You have access to customer, order, product,
    and store policy tools.

    Always use the tools when information is required.

    Never invent customer information.
    Never invent order information.
    Never invent product information.
    Never invent prices.
    Never invent tracking numbers.
    Never invent delivery dates.
    Never invent policies.
    Never invent phone numbers.
    Never invent email addresses.
    Never invent websites or support contact information.

    If required information cannot be verified
    using the available tools, clearly say that
    the information could not be verified.

    If an ID is required but missing,
    ask the customer for the required ID.
    """,

    tools=[
        customer_tool,
        order_tool,
        product_tool,
        knowledge_base_tool
    ],

    llm=llm,

    verbose=False,

    allow_delegation=False
)


# =========================================================
# TASK
# =========================================================

def create_task(question):

    return Task(

        description=f"""
        Answer the customer's question.

        Customer question:
        {question}

        Use the available tools whenever
        database or policy information is required.

        Do not invent information.

        If an ID is required but missing,
        ask the customer for it.

        Only provide information that can be
        verified from the store database
        or knowledge base.

        Do not create fake contact details.
        """,

        expected_output="""
        A clear, concise and accurate
        customer support response based only
        on verified store information.
        """,

        agent=customer_support_agent
    )


# =========================================================
# STREAMLIT CHAT
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


question = st.chat_input(
    "Ask your customer support question..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            task = create_task(question)

            crew = Crew(
                agents=[customer_support_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )

            result = crew.kickoff(
                inputs={
                    "question": question
                }
            )

            answer = str(result)

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
