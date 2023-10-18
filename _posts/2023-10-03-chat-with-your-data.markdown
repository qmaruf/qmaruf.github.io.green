---
layout: post
title: "Talk to Your Data Using LangChain and LLM"
date: 2023-10-06
description: Have you ever contemplated the possibility of engaging in a conversation with your data? Imagine conversing with a chatbot that possesses comprehensive knowledge about your dataset. This intriguing problem is the focus of this note, where we explore how to achieve it using the ChatGPT API.
---

Demo [https://qmaruf-talk-to-data.hf.space](https://qmaruf-talk-to-data.hf.space)

Have you ever contemplated the possibility of engaging in a conversation with your data? Imagine conversing with a chatbot that possesses comprehensive knowledge about your dataset. This intriguing problem is the focus of this note, where we explore how to achieve it using the ChatGPT API.

To illustrate this concept, we will employ the first book of the Harry Potter series, "[Harry Potter and the Philosopher's Stone](https://github.com/amephraim/nlp/blob/master/texts/J.%20K.%20Rowling%20-%20Harry%20Potter%201%20-%20Sorcerer's%20Stone.txt)," as our dataset. Our goal is to engage in a conversation with the content of the book. To facilitate this, we will utilize [LangChain](https://www.langchain.com/), a powerful tool for parsing and interacting with data, in conjunction with the ChatGPT API.

Our first step is to load the relevant data from the book. We will use the TextLoader from LangChain to achieve this:

```python
from langchain.document_loaders import TextLoader

book_txt = 'docs/potter1.txt'
loader = TextLoader(book_txt)
docs = loader.load()
```

Next, we need to break down the text into manageable chunks. This allows us to work with smaller sections of the book at a time. We define the chunk_size and chunk_overlap parameters for this purpose:

```python
chunk_size = 1000
chunk_overlap = 250
```

In the next step, we will create a text splitter based on `RecursiveCharacterTextSplitter`. This text splitter is ideal for generic content and uses a character parameter list for segmentation. It sequentially applies these characters to divide text into appropriately sized chunks. It ensures that paragraphs, sentences, and words stay together for maximum semantic coherence.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len
)

splits = text_splitter.split_documents(docs)
```

Here `length_function` measures the length of given chunks. We will use `len` as our `length_function` for this example.

The `split_documents` method will return a list of `Document` objects. For example, here is the first `Document` object from the list. We can print this content using the print method: `print(splits[0])`

```text
Document(page_content="Harry Potter and the Sorcerer's Stone\n\n\nCHAPTER ONE\n\nTHE 
BOY WHO LIVED\n\nMr. and Mrs. Dursley, of number four, Privet Drive, were proud to 
say\nthat they were perfectly normal, thank you very much. They were the last\npeople
you'd expect to be involved in anything strange or mysterious,\nbecause they just 
didn't hold with such nonsense.\n\nMr. Dursley was the director of a firm called 
Grunnings, which made\ndrills. He was a big, beefy man with hardly any neck, although
he did\nhave a very large mustache. Mrs. Dursley was thin and blonde and had\nnearly 
twice the usual amount of neck, which came in very useful as she\nspent so much of 
her time craning over garden fences, spying on the\nneighbors. The Dursleys had a 
small son called Dudley and in their\nopinion there was no finer boy anywhere.", 
metadata={'source': 'docs/potter1.txt'})
```

In this step, we will create a vector database to store the embeddings of the chunks. For any query, we search the vector database and extract the most similar chunks to the query. We will use Chroma vector db for this example.

```python
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
persist_directory = 'docs/chroma'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
```

Here `embedding` is the OpenAI embedding function. `persist_directory` is the directory to store the embeddings.

We can now search the vector database using `vectordb.max_marginal_relevance_search` (MMR) function. MMR returns chunks selected using the maximal marginal relevance. Maximal marginal relevance optimizes for similarity to query and diversity among selected documents. It takes the query string and returns the `k` most similar chunks. We will use `k=5` for this example.

```python
query = "Write the names of all of Harry Potter's teachers."
answers = vectordb.max_marginal_relevance_search(query, k=10)
```

Here `answers` contain the `k` most similar chunks to the query.

We can check all the chunks using a `for` loop. Here is the first answer from the list.

```
Professor Flitwick, the Charms teacher, was a tiny little wizard who had
to stand on a pile of books to see over his desk. At the start of their
first class, he took the roll call, and when he reached Harry's name, he
gave an excited squeak and toppled out of sight. Professor McGonagall was 
again different. Harry had been quite right to think she wasn't a teacher 
to cross. Strict and clever, she gave them a talking-to the moment they sat
 down in her first class.
```

Up to this point, we have created the vector database and searched the database using a query for relevant documents. Now we will use the ChatGPT API to chat with the content of the book. We will use `answers` as chat context.

Now using `langchain`, we will create a `ChatOpenAI` model to interact with the ChatGPT API.

```python
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
```

Here `model_name` is the name of the model. We will use `gpt-3.5-turbo` for this example. `temperature` will control the randomness of the chatbot response.

We will also define a `retriever` to extract the most similar chunks to the query.

```python
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 10, 'fetch_k': 50})
```

Now we will create a `RetrievalQA` chain using the `llm` and `retriever`.

```python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)
```

Here `qa_chain` combines a retriever and a language model to retrieve relevant documents for a query and answer questions based on those documents.

Let's check how `qa_chain` performs for a query. We will use the same query that we used earlier.

```python
response = qa_chain({"query": query})
print(response['result'])
```

```text
Professor Flitwick, Professor McGonagall, Professor Sprout, Professor Binns, Professor Snape, Madam Hooch
```

We can use a custom prompt to tell `qa_chain` how we want the answer. Here it is:

```python
from langchain.prompts import PromptTemplate

template = """Use only the following context to answer the question at the end. Always say "thanks for asking!" at the end of the answer. Write a summary of the question and then give the answer.
Context: {context}
Question: {question}
Answer:
{context}
Question: {question}
Answer:"""

qa_chain_prompt = PromptTemplate.from_template(template)
```

We will now fit the template into the `qa_chain` and check the result.

```python
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": qa_chain_prompt}
)
response = qa_chain({"query": query})
print(response['result'])
```

```text
The names of all of Harry Potter's teachers are Professor Flitwick, Professor McGonagall, Professor Binns, Professor Snape, Madam Hooch, and Hagrid. Thanks for asking!

```

`qa_chain` is able to understand the context of the query and give a reasonable answer.

Until now, `qa_chain` has no memory. That means we can't ask any question based on the previous answer. We will use `ConversationBufferMemory` to create a new type of chain that can remember the previous conversation. Let's define `memory` as an instance of `ConversationBufferMemory` and use it to create a new chain named `ConversationalRetrievalChain`.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)
```

Here's how to create `ConversationalRetrievalChain` using `memory`, `vectordb`, and `llm`.

```python
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(),
    memory=memory
)
```

We will ask three related questions to `qa_chain` and check the result.
```python
q1 = "Write the names of all of Harry Potters Teachers."
q2 = "sort the name of the teacher based on how frequently they are mentioned"
q3 = "tell me more about this professor"

for q in [q1, q2, q3]:
    response = qa({'question': q})
    print (f'Q: {q}')
    print (f'A: {response["answer"]}')
    print ('\n')
```


```text
Q: Write the names of all of Harry Potters Teachers.
A: The names of Harry Potter's teachers mentioned in the given context are:

1. Professor Flitwick (Charms teacher)
2. Professor McGonagall (unknown subject)
3. Professor Sprout (Herbology teacher)
4. Professor Binns (History of Magic teacher)
5. Professor Snape (Potions teacher)
6. Madam Hooch (unknown subject)
7. Professor Quirrell (unknown subject)

Please note that there may be other teachers at Hogwarts that are not mentioned in 
this context.

Q: sort the name of the teacher based on how frequently they are mentioned
A: Professor McGonagall is mentioned most frequently in the given context.

Q: tell me more about this professor
A: Professor McGonagall is described as strict and clever. She is a teacher at 
Hogwarts School of Witchcraft and Wizardry and teaches Transfiguration, which 
is described as complex and dangerous magic. She gives the students a talking-to
in her first class, emphasizing the importance of taking her class seriously. 
She is also shown to be observant and recognizes Harry's talent as a Seeker in 
Quidditch, recommending him to the Gryffindor team captain. Additionally, she 
is a member of the staff and is seen interacting with other professors, such 
as Professor Flitwick.
```

Well, it looks like `qa_chain` is able to remember the previous conversation and answer the questions based on the previous conversation.

This is the end of this note. We have seen how to use `langchain` to create a chatbot that can chat with the content of a book. We have also seen how to create a chatbot that can remember the previous conversation.




