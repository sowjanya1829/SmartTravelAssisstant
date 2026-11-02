from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

#folder path where the documents are stored
data_folder=Path("data/cities")

#configure the text splitter
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len
)

all_chunks=[]

#find every file in the data folder
for file_path in sorted(data_folder.glob("*.txt")):
    #read the file content
    text=file_path.read_text(encoding="utf-8")

    #get the city name from the file name
    city=file_path.stem.replace("_"," ").title()

    #split the text into chunks and attach the metadata
    chunks=text_splitter.create_documents(
        texts=[text],
        metadatas=[
            {
                "city":city,
                "source":str(file_path)
            }
        ]
    )
    all_chunks.extend(chunks)

    print(f"{city}:{len(chunks)} chunks created.")

print(f"Total chunks created: {len(all_chunks)}")

#display first chunk
if all_chunks:
    print("\n First chunk:")
    print(all_chunks[0].page_content)
    print("metadata: ",all_chunks[0].metadata)
else:
    print("No chunks created.")

