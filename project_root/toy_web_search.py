"""
toy_web_search.py

A toy "web search" placeholder that simulates a search by matching
keywords in a small local documents list. In a real system, you'd
have a proper search index or call a real search API.
"""

local_docs = [
    "TensorFlow is an open-source machine learning framework.",
    "You have a residual block with attention in your code.",
    "Deep learning can be used for text-to-image generation.",
    "This is a toy local document simulating a search result.",
    "Neural networks can be applied to many tasks."
]

def toy_web_search(query: str):
    """
    A very basic search that returns lines from local_docs
    containing the query (case-insensitive).
    """
    q_lower = query.lower()
    matches = [doc for doc in local_docs if q_lower in doc.lower()]
    return matches
