#utils.py
def display_fact_check_result(result):
    """Display fact-checking results in a user-friendly format"""
    print("\n===== FACT CHECK RESULT =====")
    print(f"CLAIM: {result['claim']}")
    
    # Display validity with a clear indicator
    if result['is_valid'] is True:
        validity = "✓ VALID"
    elif result['is_valid'] is False:
        validity = "✗ INVALID"
    else:
        validity = "? UNCERTAIN"
        
    print(f"VERDICT: {validity} (Confidence: {result['confidence']})")
    
    print("\nEXPLANATION:")
    print(result['explanation'])
    
    print("\nSOURCES:")
    if result['sources']:
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source.get('title', 'Untitled')} ({source.get('source', 'Unknown source')})")
            print(f"   URL: {source.get('url', 'No URL')}")
            if source.get('publication_date'):
                print(f"   Published: {source['publication_date']}")
            print()
    else:
        print("No specific sources found.")
    
    print("=============================\n")
