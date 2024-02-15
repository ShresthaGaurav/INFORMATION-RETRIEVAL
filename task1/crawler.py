import os
from datetime import datetime, timedelta
from flask import Flask, render_template, request
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import MultifieldParser
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from whoosh import scoring

app = Flask(__name__, template_folder="html")

def crawl_and_index(base_url, index_path):
    # Check the last download date
    last_download_date = get_last_download_date(index_path)

    # If last download date is not available or more than 20 days ago, re-download
    if not last_download_date or (datetime.now() - last_download_date > timedelta(days=20)):
        print("Downloading and indexing data...")

        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        schema = Schema(title=TEXT(stored=True), authors=TEXT(stored=True), year=ID(stored=True), 
                        publication_url=ID(stored=True, unique=True), author_profile_url=ID(stored=True))

        if not os.path.exists(index_path):
            os.mkdir(index_path)

        ix = create_in(index_path, schema)
        writer = ix.writer()
  

        for publication_div in soup.find_all('div', class_='result-container'):
            title_tag = publication_div.find('h3', class_="title")

            if title_tag:
                title = title_tag.get_text(strip=True)
            else:
                title = "N/A"

            authors_tags = publication_div.find_all('a', class_='link person')
            authors = [author.text.strip() for author in authors_tags] if authors_tags else ["N/A"]

            year_tag = publication_div.find('span', class_='date')
            year = year_tag.text.strip() if year_tag else "N/A"
             
            publication_url_tag = publication_div.find('a', class_='link')
            publication_url = urljoin(base_url, publication_url_tag['href']) if publication_url_tag and 'href' in publication_url_tag.attrs else "N/A"

            author_profile_url_tag = publication_div.find('a', class_='link person')
            author_profile_url = urljoin(base_url, author_profile_url_tag['href']) if author_profile_url_tag else "N/A"

            try:
                writer.add_document(title=title, authors=', '.join(authors), year=year,
                                    publication_url=publication_url, author_profile_url=author_profile_url)

            except Exception as e:
                print(f"Error adding document: {e}")

        try:
            # Commit changes to the index
            writer.commit()
            print("Indexing completed.")

            # Update the last download date
            update_last_download_date(index_path)

        except Exception as e:
            print(f"Error committing changes: {e}")

    else:
        print("Data is already up-to-date. No need to re-download.")

def get_last_download_date(index_path):
    last_download_file_path = os.path.join(index_path, "last_download_date.txt")

    try:
        with open(last_download_file_path, 'r') as file:
            last_download_date_str = file.read().strip()
            return datetime.strptime(last_download_date_str, "%Y-%m-%d")
    except FileNotFoundError:
        return None

def update_last_download_date(index_path):
    last_download_file_path = os.path.join(index_path, "last_download_date.txt")
    current_date_str = datetime.now().strftime("%Y-%m-%d")

    with open(last_download_file_path, 'w') as file:
        file.write(current_date_str)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    user_query = request.form['query']
    index_path = "DataStorage"

    ix = open_dir(index_path)
    with ix.searcher(weighting=scoring.TF_IDF()) as searcher:
         query_parser = MultifieldParser(["title", "authors"], ix.schema)
         query = query_parser.parse(user_query)
         results = searcher.search(query, terms=True)
         return render_template('search_results.html', results=results, query=user_query)
if __name__ == '__main__':
    base_url = "https://pureportal.coventry.ac.uk/en/organisations/ihw-centre-for-health-and-life-sciences-chls/publications/"
    index_path = "DataStorage"
    crawl_and_index(base_url, index_path)
    app.run(debug=True)
