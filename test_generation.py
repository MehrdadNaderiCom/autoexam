from collector import wikipedia_collector
from question_generator import question_generator

def test_question_generation():
    # Test topic
    topic = "Python programming language"
    
    print(f"Fetching content for topic: {topic}")
    content = wikipedia_collector.get_topic_content(topic)
    
    print("\nProcessing content...")
    processed_content = wikipedia_collector.process_content(content)
    
    print("\nGenerating questions...")
    questions = question_generator.generate_questions(processed_content)
    
    print("\nGenerated Questions:")
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}:")
        if question['type'] == 'fill_blank':
            print(f"Type: Fill in the blank")
            print(f"Question: {question['question']}")
            print(f"Answer: {question['answer']}")
        else:
            print(f"Type: Multiple Choice")
            print(f"Question: {question['question']}")
            print(f"Options: {', '.join(question['options'])}")
            print(f"Answer: {question['answer']}")

if __name__ == '__main__':
    test_question_generation() 