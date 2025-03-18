import wikipedia
import logging
import nltk
import os
import requests
from typing import Dict, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikipediaCollector:
    def __init__(self):
        """Initialize the WikipediaCollector."""
        logger.info("Initializing WikipediaCollector")
        wikipedia.set_lang('en')
        self.use_nltk = True
        try:
            nltk.data.find('tokenizers/punkt')
            logger.info("NLTK data found")
        except LookupError:
            logger.info("NLTK data not found, using basic text processing")
            self.use_nltk = False

    def get_topic_content(self, topic: str) -> Optional[Dict[str, str]]:
        """Get content from Wikipedia for a given topic."""
        try:
            logger.info(f"Searching Wikipedia for topic: {topic}")
            # Search for the topic
            search_results = wikipedia.search(topic)
            if not search_results:
                logger.warning(f"No Wikipedia results found for topic: {topic}")
                return None

            # Get the page for the first result
            page_title = search_results[0]
            logger.info(f"Found Wikipedia page: {page_title}")
            
            try:
                page = wikipedia.page(page_title)
            except wikipedia.exceptions.DisambiguationError as e:
                logger.info(f"Disambiguation page found, using first option: {e.options[0]}")
                page = wikipedia.page(e.options[0])
            except wikipedia.exceptions.PageError:
                logger.warning(f"Wikipedia page not found for: {page_title}")
                return None

            # Get the content and URL
            content = {
                'text': page.content,
                'url': page.url
            }
            logger.info(f"Successfully retrieved content from Wikipedia, URL: {page.url}")
            return content

        except Exception as e:
            logger.error(f"Error getting Wikipedia content: {str(e)}", exc_info=True)
            return None

    def process_content(self, content: str) -> str:
        """Process the Wikipedia content for question generation."""
        if not content:
            return ""

        # Basic content cleaning
        lines = content.split('\n')
        processed_lines = []

        for line in lines:
            # Skip empty lines and references
            if not line.strip() or line.startswith('==') or '[' in line:
                continue
            processed_lines.append(line)

        processed_content = ' '.join(processed_lines)

        # Use NLTK for sentence tokenization if available
        if self.use_nltk:
            try:
                sentences = nltk.sent_tokenize(processed_content)
                # Keep only sentences that are likely to be informative
                sentences = [s for s in sentences if len(s.split()) > 5]
                return ' '.join(sentences)
            except Exception as e:
                logger.warning(f"NLTK processing failed: {str(e)}")

        # Fallback to basic processing
        return processed_content

# Initialize the collector
wikipedia_collector = WikipediaCollector() 