import nltk
import random
from typing import List, Dict
import logging

class QuestionGenerator:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('averaged_perceptron_tagger')
        except LookupError:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
        
        logging.info("Question Generator initialized")

    def generate_multiple_choice_question(self, context: str) -> Dict:
        """Generate a multiple-choice question from the given context."""
        sentences = nltk.sent_tokenize(context)
        if not sentences:
            return None
            
        # Select a random sentence for the question
        target_sentence = random.choice(sentences)
        words = nltk.word_tokenize(target_sentence)
        pos_tags = nltk.pos_tag(words)
        
        # Find important words (nouns, verbs) to create question
        important_words = [(word, pos) for word, pos in pos_tags if pos.startswith(('NN', 'VB', 'JJ'))]
        if not important_words:
            return None
            
        # Select a random important word to create question about
        word, pos = random.choice(important_words)
        
        # Create question based on word type
        if pos.startswith('NN'):
            question = f"What {word.lower()} is mentioned in the text?"
        elif pos.startswith('VB'):
            question = f"What does the text say about {word.lower()}?"
        else:
            question = f"Which {word.lower()} is described in the text?"
        
        # Create options
        correct_answer = target_sentence
        other_sentences = [s for s in sentences if s != target_sentence]
        options = [correct_answer]
        
        # Add other sentences as distractors
        if other_sentences:
            distractors = random.sample(other_sentences, min(3, len(other_sentences)))
            options.extend(distractors)
        
        # If we don't have enough distractors, add some generic ones
        while len(options) < 4:
            options.append(f"None of the above")
        
        random.shuffle(options)
        
        return {
            'question_text': question,
            'answer_text': correct_answer,
            'options': options,
            'question_type': 'multiple_choice',
            'source_text': context
        }

    def generate_fill_blank_question(self, context: str) -> Dict:
        """Generate a fill-in-the-blank question from the given context."""
        sentences = nltk.sent_tokenize(context)
        if not sentences:
            return None
            
        # Select a random sentence
        sentence = random.choice(sentences)
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        
        # Find nouns or important words to blank out
        nouns = [(word, i) for i, (word, pos) in enumerate(pos_tags) if pos.startswith(('NN', 'VB', 'JJ'))]
        
        if not nouns:
            return None
            
        # Select a random noun to blank out
        word_to_blank, index = random.choice(nouns)
        
        # Create the fill-in-the-blank question
        words[index] = "________"
        question_text = " ".join(words)
        
        return {
            'question_text': question_text,
            'answer_text': word_to_blank,
            'question_type': 'fill_blank',
            'source_text': context
        }

    def generate_questions(self, text_chunks: List[str], num_questions: int = 10) -> List[Dict]:
        """
        Generate a mix of multiple-choice and fill-in-the-blank questions.
        """
        questions = []
        num_each_type = num_questions // 2
        
        # Generate multiple-choice questions
        for chunk in text_chunks[:num_each_type]:
            try:
                question = self.generate_multiple_choice_question(chunk)
                if question:
                    questions.append(question)
            except Exception as e:
                logging.error(f"Error generating multiple-choice question: {str(e)}")
                continue
        
        # Generate fill-in-the-blank questions
        for chunk in text_chunks[num_each_type:num_questions]:
            try:
                question = self.generate_fill_blank_question(chunk)
                if question:
                    questions.append(question)
            except Exception as e:
                logging.error(f"Error generating fill-in-blank question: {str(e)}")
                continue
        
        return questions[:num_questions]

# Initialize the question generator
question_generator = QuestionGenerator() 