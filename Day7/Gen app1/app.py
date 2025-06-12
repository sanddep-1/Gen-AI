from langchain_community.document_loaders import TextLoader
loader = TextLoader('kalam.txt',encoding='utf-8')
data = loader.load()

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    '''
    You are an AI assistant. summarize the content
    in  a format of
    Title : <Title>,
    Topic : <Topic>
    Main points : <Main points>
    context : {context}
    '''
)

from langchain_groq import ChatGroq

groq_api_key = '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'
llm = ChatGroq(groq_api_key = groq_api_key, model='llama3-70b-8192')


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitters = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 50)
splitters = text_splitters.split_documents(data)

from langchain.chains.combine_documents import create_stuff_documents_chain

chain_doc = create_stuff_documents_chain(prompt=prompt,llm=llm)

response = chain_doc.invoke({"context" : splitters})

print(response)

print("---------------------------------------------------------------")



from langchain.chains.combine_documents import create_map_reduce_documents_chain

map_prompt = ChatPromptTemplate.from_template(
    '''
    you are an AI Assistant. summarize the each and every chunk
    context : {context}
    summary :
    '''
)

reduce_prompt = ChatPromptTemplate.from_template(
    '''
    You are an AI assistant. combine all summaries of chunks.
    provide a single summary
    summary : {context} 
    Final Summary :
    '''
)

from langchain.chains.summarize import load_summarize_chain

chain_map = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",map_prompt = map_prompt, reduce_prompt = reduce_prompt
)


response = chain_map.invoke(splitters)



'''  Not working as version mistaching'''