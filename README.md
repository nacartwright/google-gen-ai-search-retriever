# Google Gen AI Search Wrapper for Langchain

This is a wrapper for Google Gen AI Search API SDK for Langchain. This implementation is configured to work with searching a website. But can easily be modified to do other types of searches.

## Overview

1. This wrapper is used to query the Google Gen AI Search API and return a list of relevant documents.
2. It then will extract the URL of the pages that were returned from the search.
3. Once it extracts the URLs it then uses the langchain WebBaseLoader to create documents out of those pages.
4. Then it chunks up the documents using the langchain text splitter.
5. Finally, it loads the documents into a FAISS vector index to return the chunks that are the cloest match.

## Limitations

The Google Gen AI Search API is currently in preview and has the following limitations:

1. While you can do a search using the user query and get a list of documents back; it does not seem possible at this time to get the actual documents back.
2. Some of the features that look to be released in the future version, such as extrative answers, are not available at this time. At least in our GCP we used during testing.

## Installation

```pip install -r requirements.txt```

## Usage

```project_id = "YOUR_PROJECT_NUMBER" # Use your project number here
search_engine_id = "YOUR_SEARCH_ENGINE_ID" # Use your search engine id here
retreiver = EnterpriseSearchRetriever(project_id=project_id, search_engine_id=search_engine_id)
docs = retreiver.get_relevant_documents("What does cdw do")

print(docs) ```