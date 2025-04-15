# main.py
import os
import sys
import json
import argparse
from rag_model import RAGFactChecker

def main():
    parser = argparse.ArgumentParser(description="Fact-check claims or articles using RAG with web scraping")
    parser.add_argument("queries", nargs="*", help="Claims or URLs to fact-check")
    parser.add_argument("--file", "-f", help="JSON file with queries to check")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    args = parser.parse_args()
    
    # Get OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set. LLM functionality will be limited.")
    
    # Set up the fact checker
    fact_checker = RAGFactChecker(api_key=api_key)
    
    # Get queries from command line or file
    queries = []
    if args.queries:
        queries = args.queries
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    queries = data
                elif isinstance(data, dict) and 'queries' in data:
                    queries = data['queries']
                else:
                    print(f"Error: Invalid format in {args.file}")
                    sys.exit(1)
        except Exception as e:
            print(f"Error reading file {args.file}: {e}")
            sys.exit(1)
    else:
        # Interactive mode
        print("Enter a claim or URL to fact-check (or 'q' to quit):")
        query = input("> ")
        while query.lower() != 'q':
            queries.append(query)
            query = input("> ")
    
    # Check all queries
    results = []
    for query in queries:
        print(f"\n--- Fact checking: {query} ---")
        result = fact_checker.fact_check(query)
        display_fact_check_result(result)
        results.append(result)
    
    # Save results if output file specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Error saving to {args.output}: {e}")

if __name__ == "__main__":
    main()