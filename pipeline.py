import kfp
import kfp.dsl as dsl

from kfp import compiler
from kfp.dsl import Dataset, Input, Output

from typing import Dict, List
import requests
from bs4 import BeautifulSoup



@dsl.component(
    base_image='python:3.11',
    packages_to_install=['appengine-python-standard']
)
def fetch_tutorial_links(url :str) -> list:
    """
    Fetches tutorial links from the given URL.

    Args:
    - url: The URL to crawl.

    Returns:
    - A list of found tutorial links.
    """
    url = 'https://pytorch.org/tutorials/'
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Raise an HTTPError for bad responses
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    soup = BeautifulSoup(response.text, 'html.parser')
    # Find all <li> tags with class 'toctree-l1'
    toc_items = soup.find_all('li', class_='toctree-l1')

    # Extract the href attribute from each <a> tag inside the <li> elements
    urls = ['https://pytorch.org/tutorials/' + item.find('a')['href'] for item in toc_items if item.find('a')]

    # Remove duplicates and return
    return list(set(urls))


@dsl.component(
    base_image='python:3.11',
    packages_to_install=['appengine-python-standard']
)
def fetch_tutorial_text(url: str) -> str:
    """
    Fetches the text content contained within <p> tags from a tutorial URL.

    Args:
    - url: The URL of the PyTorch tutorial.

    Returns:
    - A string containing the text from all <p> tags.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Raise an HTTPError for bad responses
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    # Create a BeautifulSoup object and specify the parser
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all <p> tags
    paragraphs = soup.find_all('p')
    
    # Extract text from each <p> tag
    text_content = '\n'.join(paragraph.get_text(strip=True) for paragraph in paragraphs if len(paragraph.get_text(strip=True)) > 20 )
    with open(url.split('/')[-1][:-5]+'.txt', 'w') as file:
        file.write(text_content)
    return url.split('/')[-1][:-5]+'.txt'




@dsl.component(
    base_image='python:3.11',
    packages_to_install=['google-cloud-aiplatform', 'appengine-python-standard']
)
def generate_embedding(txt_file: str) -> Dict:
    from vertexai.language_models import TextEmbeddingModel

    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

    with open(txt_file.replace("gs://", "/gcs/"), 'r') as f:
        text = f.read()
        embeddings = model.get_embeddings([text])
        embedding = embeddings[0].values

    return {"id": txt_file, "embedding": embedding}

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['elasticsearch', 'appengine-python-standard']
)
def write_embeddings(embedding: Dict):
    from elasticsearch import Elasticsearch

    # Connect to the Elasticsearch instance
    es = Elasticsearch(
        hosts=["http://34.118.227.159:9200"],
        basic_auth=("elastic", "TODO")
    )

    # Name of the index
    index_name = "technology_papers_and_reports"

    # Define the mapping for the index
    mapping = {
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768
                }
            }
        }
    }

    # Create the index with the mapping
    es.indices.create(index=index_name, body=mapping, ignore=400)

    # Index the vector embeddings
    es.index(index=index_name, id=embedding["id"], body={"embedding": embedding["embedding"]})

    print("Embeddings indexed successfully.")


@dsl.pipeline(
    name="technology-papers-and-reports",
)
def technology_papers_and_reports(url: str):
    fetch_tutorial_links_task = fetch_tutorial_links(
        url=url
    )
    with dsl.ParallelFor(
        name="pdf-parsing",
        items=fetch_tutorial_links_task.output,
        parallelism=3
    ) as pdf_file:
        fetch_tutorial_text_task = fetch_tutorial_text(
            url=pdf_file
        )
        with dsl.ParallelFor(
            name="pdf-page-parsing",
            items=fetch_tutorial_text_task.output,
            parallelism=3
        ) as pdf_page_file:
            generate_embedding_task = generate_embedding(
                txt_file=pdf_page_file
            )
            write_embeddings_task = write_embeddings(
                embedding=generate_embedding_task.output
            )


compiler.Compiler().compile(technology_papers_and_reports, 'pipeline.json')