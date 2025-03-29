import pandas as pd
import re
import os
from collections import Counter
import string

def load_and_clean_transcripts(filepath):
    """
    Load the combined transcript file and perform basic cleaning
    """
    print(f"Loading transcript file: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as file:
        raw_text = file.read()
    
    # Split by episode markers
    pattern = r"--- Episode: (.*?) ---"
    episodes = re.split(pattern, raw_text)
    
    # Create a list of episodes with titles
    episode_data = []
    for i in range(1, len(episodes), 2):
        if i+1 < len(episodes):
            title = episodes[i].strip()
            content = episodes[i+1].strip()
            episode_data.append({"title": title, "content": content})
    
    print(f"Found {len(episode_data)} episodes with titles and content")
    
    # Create DataFrame
    df = pd.DataFrame(episode_data)
    
    # Basic cleaning for each transcript
    df['clean_content'] = df['content'].apply(basic_clean)
    
    return df

def basic_clean(text):
    """
    Perform basic text cleaning
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove episode markers, timestamps, and speaker identifiers
    text = re.sub(r'\[.*?\]', '', text)  # Remove content in square brackets
    text = re.sub(r'\(.*?\)', '', text)  # Remove content in parentheses
    text = re.sub(r'\d{1,2}:\d{2}', '', text)  # Remove timestamps like 1:30, 10:45
    text = re.sub(r'^\w+:', '', text, flags=re.MULTILINE)  # Remove speaker names like "Robert:" at start of lines
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove punctuation except apostrophes (to keep contractions)
    punctuation_to_remove = string.punctuation.replace("'", "")
    translator = str.maketrans('', '', punctuation_to_remove)
    text = text.translate(translator)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_basic_stats(df):
    """
    Calculate basic statistics about the transcripts
    """
    # Word counts
    df['word_count'] = df['clean_content'].apply(lambda x: len(x.split()))
    
    # Basic stats
    total_words = df['word_count'].sum()
    avg_words = df['word_count'].mean()
    min_words = df['word_count'].min()
    max_words = df['word_count'].max()
    
    print("\n--- BASIC STATISTICS ---")
    print(f"Total episodes: {len(df)}")
    print(f"Total words: {total_words}")
    print(f"Average words per episode: {avg_words:.2f}")
    print(f"Shortest episode: {min_words} words")
    print(f"Longest episode: {max_words} words")
    
    return {
        'total_episodes': len(df),
        'total_words': total_words,
        'avg_words': avg_words,
        'min_words': min_words,
        'max_words': max_words
    }

def analyze_word_frequencies_simple(df, top_n=50):
    """
    Analyze word frequencies without using NLTK tokenization
    """
    # Define stopwords manually
    stopwords = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when',
        'where', 'how', 'why', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
        'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
        'does', 'did', 'doing', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'can', 'will', 'should', 'now', 'of', 'at',
        'by', 'ive', 'im', "i'm", 'youre', 'dont', "it’s", 'its', 'id', 'youve', 'didnt', 'isnt', 'thats',
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'um', 'uh', 'yeah', 'like', 'know', 'think', 'going', 'thing', 'really',
        'lot', 'sort', 'kind', 'just', 'right', 'way', 'well', 'mean', 'actually',
        'probably', 'okay', 'yes', 'sure', 'good', 'great', 'listeners', 'podcast',
        '80000', 'hours', 'episode', 'transcript', 'get', 'got', 'getting', "we're", "don't"
        "that's", "there's", "i've", "i'd", "i'll", "you're", "you've", "you'll", "he's", "she's", "it's",
        "we've", "we'll", "they're", "they've", "they'll", "isn't", "aren't", "wasn't", "weren't", "haven't",
        "hasn't", "hadn't", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot", "couldn't",
        "i'm", "you're", "he's", "she's", "it's", "we're", "they're", "i've", "you've", "we've", "they've",
        'rob', 'wiblin', "that's", "would", 'rob wiblin', 'robert wiblin', 'luisa rodriguez', 'arden koehler',
        'holden karnofsky', "that's one", 'robert'
    }
    
    # Combine all cleaned text
    all_text = ' '.join(df['clean_content'])
    
    # Simple tokenization by splitting on spaces
    words = all_text.split()
    
    # Filter out stopwords and short words
    filtered_words = [word for word in words if word.lower() not in stopwords and len(word) > 2]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Get top words
    top_words = word_counts.most_common(top_n)
    
    print("\n--- TOP WORDS ---")
    for word, count in top_words[:20]:
        print(f"{word}: {count}")
    
    return top_words

def extract_ngrams_simple(df, n=2, top_n=30):
    """
    Extract most common n-grams without using NLTK
    """
    # Define stopwords manually (same as above)
    stopwords = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when',
        'where', 'how', 'why', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
        'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
        'does', 'did', 'doing', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'can', 'will', 'should', 'now', 'of', 'at',
        'by', 'ive', 'im', "i'm", 'youre', 'dont', "it’s", 'its', 'id', 'youve', 'didnt', 'isnt', 'thats',
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'um', 'uh', 'yeah', 'like', 'know', 'think', 'going', 'thing', 'really',
        'lot', 'sort', 'kind', 'just', 'right', 'way', 'well', 'mean', 'actually',
        'probably', 'okay', 'yes', 'sure', 'good', 'great', 'listeners', 'podcast',
        '80000', 'hours', 'episode', 'transcript', 'get', 'got', 'getting', "we're", "don't",
        "that's", "there's", "i've", "i'd", "i'll", "you're", "you've", "you'll", "he's", "she's", "it's",
        "we've", "we'll", "they're", "they've", "they'll", "isn't", "aren't", "wasn't", "weren't", "haven't",
        "hasn't", "hadn't", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot", "couldn't",
        "i'm", "you're", "he's", "she's", "it's", "we're", "they're", "i've", "you've", "we've", "they've",
        'rob', 'wiblin', "that's", "would", 'rob wiblin', 'robert wiblin', 'luisa rodriguez', 'arden koehler',
        'holden karnofsky', "that's one", 'robert', 'dave denkenberger'
    }
    
    # Function to extract n-grams from text
    def get_ngrams(text, n):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stopwords and len(word) > 2]
        ngrams = [' '.join(filtered_words[i:i+n]) for i in range(len(filtered_words)-n+1)]
        return ngrams
    
    # Get n-grams from all texts
    all_ngrams = []
    for text in df['clean_content']:
        all_ngrams.extend(get_ngrams(text, n))
    
    # Count frequencies
    ngram_freq = Counter(all_ngrams)
    
    # Get top n-grams
    top_ngrams = ngram_freq.most_common(top_n)
    
    print(f"\n--- TOP {n}-GRAMS ---")
    for ngram, count in top_ngrams[:20]:
        print(f"{ngram}: {count}")
    
    return top_ngrams

def save_clean_transcripts(df, output_folder):
    """
    Save cleaned transcripts to files
    """
    # Create output folder if it doesn't exist
    clean_folder = os.path.join(output_folder, "cleaned")
    if not os.path.exists(clean_folder):
        os.makedirs(clean_folder)
    
    # Save each cleaned transcript
    for i, row in df.iterrows():
        # Create a safe filename
        safe_title = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in row['title'])
        filename = f"{clean_folder}/{safe_title}_clean.txt"
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(row['clean_content'])
    
    # Save combined cleaned transcript
    combined_filename = f"{clean_folder}/80000hours_podcast_transcripts_combined_clean.txt"
    with open(combined_filename, 'w', encoding='utf-8') as f:
        for i, row in df.iterrows():
            f.write(f"--- Episode: {row['title']} ---\n\n")
            f.write(row['clean_content'])
            f.write("\n\n")
    
    print(f"\nCleaned transcripts saved to {clean_folder}")
    print(f"Combined cleaned transcript saved to {combined_filename}")
    
    return combined_filename

def generate_simple_report(stats, top_words, top_bigrams, output_folder):
    """
    Generate a simple text report with analysis results
    """
    report_path = os.path.join(output_folder, "transcript_analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("80,000 HOURS PODCAST TRANSCRIPT ANALYSIS\n")
        f.write("=======================================\n\n")
        
        # Basic stats
        f.write("BASIC STATISTICS\n")
        f.write("-----------------\n")
        f.write(f"Total episodes: {stats['total_episodes']}\n")
        f.write(f"Total words: {stats['total_words']}\n")
        f.write(f"Average words per episode: {stats['avg_words']:.2f}\n")
        f.write(f"Shortest episode: {stats['min_words']} words\n")
        f.write(f"Longest episode: {stats['max_words']} words\n\n")
        
        # Top words
        f.write("TOP 50 WORDS\n")
        f.write("-----------\n")
        for i, (word, count) in enumerate(top_words, 1):
            f.write(f"{i}. {word}: {count}\n")
        f.write("\n")
        
        # Top bigrams
        f.write("TOP 30 BIGRAMS (WORD PAIRS)\n")
        f.write("--------------------------\n")
        for i, (bigram, count) in enumerate(top_bigrams, 1):
            f.write(f"{i}. {bigram}: {count}\n")
    
    print(f"\nAnalysis report saved to {report_path}")
    return report_path

def main(transcript_filepath, output_folder):
    """
    Main function to run the analysis pipeline
    """
    # Load and clean data
    df = load_and_clean_transcripts(transcript_filepath)
    
    # Analyze basic stats
    stats = analyze_basic_stats(df)
    
    # Analyze word frequencies (without NLTK)
    top_words = analyze_word_frequencies_simple(df, top_n=50)
    
    # Extract common bigrams (without NLTK)
    top_bigrams = extract_ngrams_simple(df, n=2, top_n=30)
    
    # Save cleaned transcripts
    save_clean_transcripts(df, output_folder)
    
    # Generate report
    generate_simple_report(stats, top_words, top_bigrams, output_folder)
    
    print("\nAnalysis complete!")
    return df

if __name__ == "__main__":
    # Replace with your transcript file path
    transcript_filepath = "80000hours_transcripts/80000hours_podcast_transcripts_combined.txt"
    output_folder = "80000hours_transcripts"
    main(transcript_filepath, output_folder)