import os
from dotenv import load_dotenv
from search import SearchEngine

def main():
    load_dotenv()
    
    try:
        engine = SearchEngine()
        print("🤖 Document Q&A System")
        print("Type 'exit' or 'quit' to end the session\n")
        
        while True:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() in ('exit', 'quit'):
                print("\n👋 Goodbye!")
                break
                
            if not query:
                continue
                
            print("\n🔍 Searching...")
            docs = engine.search_similar(query)
            
            print("💭 Thinking...")
            response = engine.get_response(query, docs)
            
            print(f"\n🤖 Answer: {response}\n")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())