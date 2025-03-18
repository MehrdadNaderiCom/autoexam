import wikipediaapi
import nltk
from typing import Dict, List, Optional

class WikipediaCollector:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='AutoExam/1.0 (mehrdadnaderiom@gmail.com)'
        )
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def get_topic_content(self, topic: str) -> Optional[Dict[str, str]]:
        """
        Fetch content from Wikipedia for a given topic.
        Returns a dictionary with title and sections if found, None otherwise.
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
        
        # Get content from sections (excluding references, external links, etc.)
        for section in page.sections:
            if not any(x in section.title.lower() for x in ['references', 'external links', 'see also', 'notes']):
                content['sections'][section.title] = section.text
                
        return content

    def process_content(self, content: Dict[str, str]) -> List[str]:
        """
        Process the Wikipedia content into chunks suitable for question generation.
        Returns a list of text chunks.
        """
        chunks = []
        
        # Add summary
        if content['summary']:
            chunks.extend(nltk.sent_tokenize(content['summary']))
        
        # Process each section
        for section_title, section_text in content['sections'].items():
            if section_text:
                # Add section sentences
                sentences = nltk.sent_tokenize(section_text)
                chunks.extend(sentences)
        
        # Filter out short sentences and clean chunks
        chunks = [chunk.strip() for chunk in chunks if len(chunk.split()) >= 10]
        
        return chunks[:20]  # Limit to 20 chunks to avoid overwhelming the model

# Initialize the collector
wikipedia_collector = WikipediaCollector() 