from embedding import DocumentEmbedder


def main() -> None:
    
    chunk_factory  = DocumentEmbedder("./news_articles")
    chunks = chunk_factory.get_chunks()



if __name__ == "__main__":
    main()


