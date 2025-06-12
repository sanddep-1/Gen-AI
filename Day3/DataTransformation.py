from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader_txt = TextLoader('pk_speech.txt', encoding='utf-8')
docs = loader_txt.load()
print(type(docs))
print(len(docs))

text_splits = RecursiveCharacterTextSplitter(chunk_size = 350, chunk_overlap = 50)

splits = text_splits.split_documents(docs)
print(splits)
print(len(splits))
print(type(splits[0]))
print(type(splits))
print(splits[0])

print("-------------------------------------------------------------------------------------")


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter


loader_txt = TextLoader('pk_speech.txt', encoding='utf-8')
docs = loader_txt.load()
print(type(docs))
print(len(docs))

text_splits = CharacterTextSplitter(chunk_size = 350, chunk_overlap = 50)

splits = text_splits.split_documents(docs)
print(splits)
print("Length ->",len(splits))
print(type(splits[0]))
print(type(splits))
print(splits[0].page_content)


print("-------------------------------------------------------------------------------------")

from langchain_text_splitters import HTMLHeaderTextSplitter

html_string = '''
<html>
<head><title>asian paints </title></head>
<style>

footer{
    position: relative; /* Needed for absolute positioning of child */
    bottom: -170; /* Stick to the bottom */
    width: 100%; /* Full width */
    text-align: center; /* Center the text */
     /* Optional: Add a background color */
}
    
body{
    background-color : Grey;
    background-image: url("C:/Users/saandeep.gunana/Desktop/Krish-Naik/Day-7-8-9/Asianpaints/static/asian_paints.jpg"); /* Or .png, .gif, etc. */
    background-size: cover; /* or contain, auto, 100% 100%, etc. - see explanation below */
    background-repeat: no-repeat; /* To prevent image tiling */
    background-attachment: fixed; /* Optional: Keeps the background fixed on scroll */
    margin: 0; /* Remove default body margins */
}


</style>

<body style="text-align: center;">
<h2> Welcome to Asian Paints</h2>
<b>
<p>Asian Paints is a leading paint company in India and one of the largest in Asia.<p>
<br>Main Leading factors of Asian paints are:</br>
<br>

<div style="text-align: center;">  <ul style="list-style-type: circle; text-align: left; display: inline-block; vertical-align: middle;">
    <li>Wide Range of Products</li>
    <li>Focus on Innovation</li>
    <li>Extensive Color Palette</li>
    <li>High Quality and Durability</li>
    <li>Strong Brand Recognition and Trust</li>
    <li>Customer-centric Approach</li>
    <li>Wide Distribution Network</li>
    <li>Sustainability Initiatives</li>
  </ul>

</div>

<div style="text-align: center;">
  <table border="1" bgcolor="Grey" style="display: inline-block; vertical-align: middle;">
    <tr>
      <td><a href = "/login">ADMIN</a></td>
      <td><a href = "/user">USER</a></td>
      <br>
      <td><a href = "/feedback">Feedback</a></td></br>
    </tr>
  </table>
</div>

<footer>
    <p> @ Copyrights are reserved only to Asian Paints as per year 2025</p>
</footer>

<h1> ASIAN PAINTS Testing</h1>

</body>
</html>'''

headers_to_split_on = [('h1', 'header1'), ('h2', 'Header2'), ('h3', 'Header3')]

text_splits = HTMLHeaderTextSplitter(headers_to_split_on)
splits = text_splits.split_text(html_string)

print(splits)
print(len(splits))


print("-------------------------------------------------------------------------------------")


import json
import requests

url = 'https://api.open-meteo.com/v1/forecast?latitude=35&longitude=139&hourly=temperature_2m'
data = requests.get(url).json()

print(data)

from langchain_text_splitters import RecursiveJsonSplitter

json_split = RecursiveJsonSplitter(max_chunk_size=30, min_chunk_size=5)

splits = json_split.split_json(data)

print(splits)

print(type(splits))
print(len(splits))
print(splits[0])