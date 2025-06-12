from langchain_community.document_loaders import TextLoader

loader_txt = TextLoader('pk_speech.txt',encoding='utf-8')
documents_txt = loader_txt.load()

print(documents_txt)

print(type(documents_txt))
print(type(documents_txt[0]))
print(len(documents_txt))

print("--------------------------------------------------------------------------------------------------")

from langchain_community.document_loaders import PyPDFLoader

loader_pdf = PyPDFLoader('Rohith Ravi Teja 18-232 Resume.pdf')
documents_pdf = loader_pdf.load()

print(documents_pdf)
print(type(documents_pdf))
print(type(documents_pdf[0]))
print(len(documents_pdf))

print("--------------------------------------------------------------------------------------------------")


import os
from langchain_community.document_loaders import WebBaseLoader

loader_web = WebBaseLoader('https://www.bing.com/search?filters=ufn%3a%22Pawan+Kalyan%22+sid%3a%223ebca3fe-81a4-a6c4-3cd0-3232fd006fa3%22&qs=MB&pq=pawa&sk=CSYN1LS12AS1UAS1&sc=19-4&pglt=41&q=pawan+kalyan&cvid=33c527b5c59d410594eeaaf808734f2b&gs_lcrp=EgRlZGdlKgYIARAuGEAyBggAEEUYOTIGCAEQLhhAMgYIAhAAGEAyBggDEEUYOzIGCAQQRRg7MgYIBRAuGEAyBggGEAAYQDIGCAcQRRg8MgYICBBFGEEyCAgJEOkHGPxV0gEIMjc0NGowajGoAgCwAgA&FORM=ANNAB1&PC=U531')

documents_web = loader_web.load()

print(documents_web)
print(type(documents_web))
print(type(documents_web[0]))
print(len(documents_web))



print("----------------------------------------------------------------------------------------------")


from langchain_community.document_loaders import WikipediaLoader

loader_wiki = WikipediaLoader("Pawan Kalyan")

documents_wiki = loader_wiki.load()

print(documents_wiki)

print("----------------------------------------------------------------------------------------------")
