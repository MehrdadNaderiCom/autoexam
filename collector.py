import wikipediaapi
import nltk
import logging
import os
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

    def get_topic_content(self, topic: str) -> Dict[str, str]:
        """
        Fetch content for a given topic from Wikipedia.
        
        Args:
            topic (str): The topic to search for
            
        Returns:
            Dict[str, str]: Dictionary containing title, summary, and sections
        """
        page = self.wiki.page(topic)
        if not page.exists():
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