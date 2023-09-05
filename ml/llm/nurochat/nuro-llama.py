import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


from llama_index import TreeIndex, SimpleDirectoryReader
from llama_index import VectorStoreIndex

# from IPython.display import Markdown, display

documents = SimpleDirectoryReader("data-nuro").load_data()
# new_index = TreeIndex.from_documents(documents)
new_index = VectorStoreIndex.from_documents(documents)


# set Logging to DEBUG for more detailed outputs

query_engine = new_index.as_query_engine()
while True:
    query = input("Enter your question: ")
    if query == "exit":
        break
    # response = query_engine.query("What did the narrator do after getting back to Chicago?")
    response = query_engine.query(query)
    print(response)
