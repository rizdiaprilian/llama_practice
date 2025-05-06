from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ollama import Client
from ollama import ChatResponse
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.prompts import PromptTemplate
from operator import itemgetter

import os

MODEL = 'llama3.2'

model = OllamaLLM(model=MODEL)
client = Client(host='http://localhost:11434')

def retrieve_information(input_prompt, persist_directory, similarity_threshold=0.7):
    """
    Retrieve relevant information from ChromaDB based on the user's prompt.
    Only returns an answer if the similarity score exceeds the threshold.
    """

    metadata_field_info = [
        AttributeInfo(
            name="any random facts about fufufufafafu",
            description="The collection of facts about fufufufafafu",
            type="string or list[string]",
        ),
        ]


    # Load the persisted ChromaDB
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=OllamaEmbeddings(model="llama3.2")
    )

    retriever = vector_store.as_retriever()

    # Initialize the output parser
    parser = StrOutputParser()

    # This example only specifies a relevant query
    # print(retriever.invoke(input_prompt))

    template = """
        Please answer the question based on the context below. If it's deemed too complex
        or not relevant, please reply "I afraid my capability has yet to satisfy to the topic you're asking"

        Context: {context}

        Question: {question}
    """

    prompt_template = PromptTemplate.from_template(template)

    # Create the chain
    chain = (
        {
        "context": itemgetter("question") | retriever, "question": itemgetter("question")
        }
        | prompt_template
        | model
        | parser
    )

    response = chain.invoke({
        "context": "We will have the chat about any facts about fufufufafafu",
        "question": input_prompt})
    
    print(response)


def main():
    print("Welcome to the Information Retrieval System!")
    print("Type 'exit' to quit the application.\n")
    chroma_dir = os.path.join(os.getcwd(), "chroma_db")
    
    while True:
        # Get user input
        user_input = input("Enter your query: ")
        
        # Exit the loop if the user types 'exit'
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Retrieve relevant information for the user's query
        retrieve_information(user_input, chroma_dir, 0.7)

if __name__ == "__main__":
    main()