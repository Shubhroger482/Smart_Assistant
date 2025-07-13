from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size=800, chunk_overlap=100) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[".", "\n", " ", ""]
    )
    return splitter.split_text(text)
