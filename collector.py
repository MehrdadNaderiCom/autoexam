import wikipediaapi
import nltk
import logging
import os
import requests
from typing import Dict, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikipediaCollector:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='AutoExam/1.0 (mnade@example.com)'
        )
        
        # Ensure NLTK data directory exists
        nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        # Download required NLTK data
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded."""
        try:
            nltk.data.find('tokenizers/punkt')
            logger.info("Found existing punkt data")
        except LookupError:
            logger.info("Downloading punkt...")
            nltk.download('punkt', quiet=True)
            logger.info("Successfully downloaded punkt")

    def _search_wikipedia(self, query: str) -> List[str]:
        """
        Search Wikipedia for a query and return a list of potential page titles.
        """
        try:
            # Use Wikipedia's API to search for pages
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": 5
            }
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract page titles from search results
            search_results = data.get("query", {}).get("search", [])
            return [result["title"] for result in search_results]
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {str(e)}")
            return []

    def get_topic_content(self, topic: str) -> Dict[str, str]:
        """
        Fetch content for a given topic from Wikipedia.
        
        Args:
            topic (str): The topic to search for
            
        Returns:
            Dict[str, str]: Dictionary containing title, summary, and sections
        """
        # Try exact match first
        page = self.wiki.page(topic)
        if not page.exists():
            # If exact match fails, try searching
            logger.info(f"No exact match found for '{topic}', searching alternatives...")
            search_results = self._search_wikipedia(topic)
            
            # Try variations of the topic
            variations = [
                topic,
                topic.lower(),
                topic.title(),
                f"{topic} (grammar)" if "tense" in topic.lower() else topic,
                f"{topic} (linguistics)" if "tense" in topic.lower() else topic
            ]
            
            # Add search results to variations
            variations.extend(search_results)
            
            # Try each variation
            for variation in variations:
                logger.info(f"Trying variation: {variation}")
                page = self.wiki.page(variation)
                if page.exists() and len(page.summary) > 100:  # Ensure we have meaningful content
                    logger.info(f"Found content using: {variation}")
                    break
            else:
                logger.error(f"Could not find suitable content for '{topic}'")
                return None
        
        # Get main content and sections
        content = {
            'title': page.title,
            'summary': page.summary,
            'sections': {}
        }
        
        # Add section content, excluding references and external links
        for section in page.sections:
            if not any(x in section.title.lower() for x in ['references', 'external links', 'see also']):
                content['sections'][section.title] = section.text
        
        return content

    def process_content(self, content: Dict[str, str]) -> str:
        """
        Process the Wikipedia content into a format suitable for question generation.
        
        Args:
            content (Dict[str, str]): The Wikipedia content
            
        Returns:
            str: Processed content as a single string
        """
        if not content:
            return ""
        
        # Combine summary and section content
        processed_text = content['summary'] + "\n\n"
        
        for section_title, section_text in content['sections'].items():
            if section_text.strip():  # Only add non-empty sections
                processed_text += f"{section_title}\n{section_text}\n\n"
        
        return processed_text.strip()

# Initialize the collector
wikipedia_collector = WikipediaCollector() 