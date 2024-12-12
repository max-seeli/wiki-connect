import requests
from warnings import warn
from typing import List, Dict

def get_page_infos(titles: List[str]) -> Dict[str, Dict]:
    """
    Fetch information about given Wikipedia pages. This information includes:
    - The title of the page (might be different from the input title due to redirects)
    - The categories the page belongs to
    - The links present in the page
    - The introductory text of the page

    Parameters
    ----------
    titles : List[str]
        The titles of the pages to fetch information about.

    Returns
    -------
    Dict[str, Dict]
        A dictionary mapping titles to dictionaries containing the fetched information.
    """
    url = "https://en.wikipedia.org/w/api.php"
    titles_info = {}
    
    for title in titles:
        params = {
            "action": "query",
            "format": "json",
            "prop": "links|categories|extracts",
            "titles": title,
            "pllimit": "max",
            "exintro": True,
            "explaintext": True,
            "redirects": True
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                warn(f"Failed to fetch links for {title}: Status code {response.status_code} - {response.text}")
                continue
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
        except Exception as e:
            warn(f"Failed to fetch links for {title}: {e}")
            continue
        for _, page_data in pages.items():
            titles_info[title] = {
                # Note: This title might be different from the input title due to redirects
                "title": page_data.get("title", ""), 
                "categories": [category["title"] for category in page_data.get("categories", [])],
                "links": [link["title"] for link in page_data.get("links", [])],
                "info_text": page_data.get("extract", "")
            }
            
        # print(f"Fetched {sum(len(v["links"]) for v in titles_info.values())} links for {len(titles_info)} pages")
    return titles_info