import os
import re
from collections import defaultdict

# Define the themes and keywords
themes = {
    "Global Health": ["global health", "disease", "poverty", "healthcare", "malaria", "developing countries", 
                     "global burden of disease", "infectious diseases", "pandemic preparedness", "public health", 
                     "health equity", "immunization", "maternal health", "child mortality", "nutrition", 
                     "sanitation", "water access", "preventive medicine"],
    
    "AI Safety": ["artificial intelligence", "ai alignment", "ai safety", "machine learning", "superintelligence", 
                 "ai risk", "artificial general intelligence", "deep learning", "neural networks", "bias in AI", 
                 "explainable AI", "robust AI", "controllable AI", "AI ethics", "value alignment", 
                 "adversarial attacks", "autonomous weapons systems"],
    
    "Longtermism": ["longterm", "future generations", "existential risk", "longterm future", "climate change", 
                   "biosecurity", "nuclear proliferation", "global catastrophic risks", "pandemic risks", 
                   "nanotechnology risks", "resource depletion", "environmental sustainability", 
                   "intergenerational equity", "future wellbeing", "planetary health", "sustainable development goals"],
    
    "Animal Welfare": ["animal welfare", "animal suffering", "factory farming", "animal rights", "veganism", 
                      "vegetarianism", "animal agriculture", "sentience", "animal cognition", 
                      "laboratory animal welfare", "wild animal welfare", "speciesism", "animal liberation"],
    
    "Poverty Reduction": ["poverty", "global poverty", "developing countries", "economic development", "inequality", 
                         "extreme poverty", "microfinance", "economic growth", "sustainable development", 
                         "poverty traps", "basic income", "wealth distribution", "social safety nets", 
                         "access to education", "access to resources"],
    
    "Career Choices": ["career", "impactful career", "job", "work", "philanthropy", "effective altruism career", 
                      "nonprofit sector", "social entrepreneurship", "impact investing", "volunteer work", 
                      "skill development", "career advice", "career paths", "leadership development", 
                      "organizational effectiveness"],
    
    "Effective Altruism": ["effective altruism", "doing good better", "moral philosophy", "consequentialism", 
                          "utilitarianism", "moral uncertainty", "cause prioritization", "cost-effectiveness", 
                          "impact evaluation", "donation effectiveness", "global priorities", "earning to give", 
                          "evidence-based giving", "charity evaluation", "counterfactual reasoning", 
                          "expected value", "moral circle expansion"]
}

def extract_passages(file_path, themes_dict, context_words=150, min_passage_length=50):
    """
    Extract passages from transcript that contain theme keywords.
    
    Args:
        file_path: Path to the transcript file
        themes_dict: Dictionary of themes and their keywords
        context_words: Number of words to include before and after keyword match
        min_passage_length: Minimum length of passage to include
        
    Returns:
        Dictionary with themes as keys and lists of passages as values
    """
    try:
        # Read the transcript file
        with open(file_path, 'r', encoding='utf-8') as file:
            transcript = file.read()
        
        # Split into words for context extraction
        words = transcript.split()
        word_count = len(words)
        
        # Dictionary to store passages for each theme
        theme_passages = defaultdict(list)
        
        # Process each theme and its keywords
        for theme, keywords in themes_dict.items():
            print(f"Processing theme: {theme}")
            
            for keyword in keywords:
                # Convert to lowercase for case-insensitive matching
                keyword_lower = keyword.lower()
                
                # Use regex to find all occurrences of the keyword
                matches = list(re.finditer(r'\b' + re.escape(keyword_lower) + r'\b', transcript.lower()))
                
                for match in matches:
                    # Get the position of the match
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Convert position to word index (approximate)
                    # This is a rough estimate that works for most texts
                    start_words = transcript[:start_pos].count(' ')
                    
                    # Extract context before and after
                    start_word_idx = max(0, start_words - context_words)
                    end_word_idx = min(word_count, start_words + context_words)
                    
                    # Extract the passage
                    passage = ' '.join(words[start_word_idx:end_word_idx])
                    
                    # Only include passages of sufficient length
                    if len(passage) >= min_passage_length:
                        # Add the passage and keyword to the theme
                        theme_passages[theme].append({
                            'passage': passage,
                            'keyword': keyword,
                            'position': start_pos
                        })
        
        # Remove duplicates and nearby passages that likely overlap
        cleaned_passages = clean_passages(theme_passages)
        return cleaned_passages
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return {}

def clean_passages(theme_passages, proximity_threshold=1000):
    """
    Remove duplicate or overlapping passages.
    
    Args:
        theme_passages: Dictionary with themes and their passages
        proximity_threshold: Character distance to consider passages as overlapping
        
    Returns:
        Cleaned dictionary of theme passages
    """
    cleaned = defaultdict(list)
    
    for theme, passages in theme_passages.items():
        # Sort passages by position
        sorted_passages = sorted(passages, key=lambda x: x['position'])
        
        if not sorted_passages:
            continue
            
        # Keep the first passage
        current = sorted_passages[0]
        cleaned[theme].append(current)
        
        # Check the rest of the passages
        for passage in sorted_passages[1:]:
            # If this passage is far enough from the last one we kept
            if passage['position'] - current['position'] > proximity_threshold:
                cleaned[theme].append(passage)
                current = passage
    
    return cleaned

def save_passages(passages_dict, output_dir):
    """
    Save passages to files organized by theme.
    
    Args:
        passages_dict: Dictionary with themes and their passages
        output_dir: Directory to save the files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each theme's passages to a separate file
    for theme, passages in passages_dict.items():
        # Create a filename from the theme
        filename = os.path.join(output_dir, f"{theme.replace(' ', '_')}_passages.txt")
        
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"=== {theme} Passages ===\n\n")
            
            for i, passage_data in enumerate(passages, 1):
                passage = passage_data['passage']
                keyword = passage_data['keyword']
                
                file.write(f"Passage {i} (Keyword: {keyword}):\n")
                file.write(f"{passage}\n")
                file.write("-" * 80 + "\n\n")
            
            file.write(f"Total passages: {len(passages)}\n")
    
    print(f"Saved passages to {output_dir}")

def main():
    # Define file paths
    transcript_file = "/Users/christianglass/Documents/GitHub/ai-governance-experiment/training/rawtrainingdata/80000hours_transcripts/cleaned/80000hours_podcast_transcripts_combined.txt"
    output_directory = "theme_passages"
    
    # Extract passages
    print(f"Extracting passages from {transcript_file}...")
    passages = extract_passages(transcript_file, themes)
    
    # Save passages to files
    save_passages(passages, output_directory)
    
    # Print summary
    total_passages = sum(len(p) for p in passages.values())
    print(f"\nExtraction complete. Found {total_passages} total passages across {len(passages)} themes.")
    for theme, theme_passages in passages.items():
        print(f"  - {theme}: {len(theme_passages)} passages")

if __name__ == "__main__":
    main()