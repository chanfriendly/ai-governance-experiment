import requests
from bs4 import BeautifulSoup
import os
import time
import bs4
import random
from datetime import datetime

# Multiple User-Agent headers to rotate through
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
]

def get_random_headers():
    """Generate random headers to mimic different browsers"""
    return {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }

def fetch_with_retry(url, max_retries=5, initial_delay=3):
    """Fetch a URL with exponential backoff retry logic"""
    retry_delay = initial_delay
    
    for retry in range(max_retries):
        try:
            headers = get_random_headers()
            print(f"  Attempt {retry+1}/{max_retries} with User-Agent: {headers['User-Agent'][:30]}...")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403 and retry < max_retries - 1:
                print(f"  Got 403 error, waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if retry < max_retries - 1:
                print(f"  Connection error: {e}, waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise
    
    raise Exception(f"Failed to fetch {url} after {max_retries} attempts")

def download_80000_hours_transcripts_combined(output_folder="80000hours_transcripts"):
    """
    Downloads transcripts of 80,000 Hours podcast episodes and combines them.
    Uses improved error handling and throttling to avoid 403 errors.
    """
    episodes_list_url = "https://80000hours.org/podcast/episodes/"
    combined_transcript_text = ""

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Track progress to enable resuming
    progress_file = f"{output_folder}/progress.txt"
    last_processed_index = 0
    
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            try:
                last_processed_index = int(f.read().strip())
                print(f"Resuming from episode index {last_processed_index}")
            except ValueError:
                print("Invalid progress file, starting from the beginning")
    
    # Log file for detailed debugging
    log_file = f"{output_folder}/scraping_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def log_message(message):
        """Log message to both console and log file"""
        print(message)
        with open(log_file, 'a', encoding='utf-8') as logf:
            logf.write(f"{message}\n")
    
    log_message(f"Starting transcript download at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Fetching episodes list from: {episodes_list_url}")
    
    try:
        # 1. Fetch the Episodes List Page with retry logic
        episodes_list_response = fetch_with_retry(episodes_list_url)
        episodes_list_soup = BeautifulSoup(episodes_list_response.content, 'html.parser')
        
        # 2. Extract Episode Links
        episode_links = []
        episode_containers = episodes_list_soup.find_all('div', class_='col-sm-9 col-md-9 navigation-list-everything__col-content-title')
        
        if not episode_containers:
            log_message("Warning: Could not find episode containers on the episodes list page. Website structure may have changed.")
            # Try an alternative selector
            episode_containers = episodes_list_soup.select('div.navigation-list-everything__col-content-title')
            if not episode_containers:
                log_message("Error: Could not find episode links with alternative selector either. Exiting.")
                return
        
        for container in episode_containers:
            link_tag = container.find('a')
            if link_tag and 'href' in link_tag.attrs:
                episode_url = link_tag['href']
                episode_links.append(episode_url)
        
        log_message(f"Found {len(episode_links)} episode links.")
        
        # If no episodes were found, something is wrong
        if not episode_links:
            log_message("Error: No episode links extracted. Website structure may have changed.")
            return
            
        # Process each episode link, starting from where we left off
        for i, episode_url in enumerate(episode_links[last_processed_index:], start=last_processed_index):
            try:
                log_message(f"\nProcessing episode {i+1}/{len(episode_links)}: {episode_url}")
                
                # Add variable delay between requests (3-7 seconds) to appear more human-like
                delay = random.uniform(3, 7)
                log_message(f"  Waiting {delay:.2f} seconds before fetching...")
                time.sleep(delay)
                
                # Fetch Episode Page with retry logic
                episode_response = fetch_with_retry(episode_url)
                episode_soup = BeautifulSoup(episode_response.content, 'html.parser')
                
                # Extract Episode Title
                episode_title_element = episode_soup.find('h1', class_='entry-title')
                episode_title = episode_title_element.get_text(strip=True) if episode_title_element else f"Unknown Episode {i+1}"
                log_message(f"  Episode title: {episode_title}")
                
                # Look for transcript using different methods
                transcript_container = None
                
                # Method 1: Try to find Transcript using HTML comment
                transcript_comment = episode_soup.find(string=lambda text: isinstance(text, bs4.Comment) and "Transcript" in text)
                if transcript_comment:
                    log_message(f"  Found Transcript HTML comment")
                    # Look for the transcript container after the comment
                    transcript_container = transcript_comment.find_next('div', class_='wrap-podcast-transcript')
                    if not transcript_container:
                        # Try to find any div that might be a container
                        transcript_container = transcript_comment.find_next('div')
                        log_message(f"  Found general div after comment: {transcript_container.get('class') if transcript_container else 'None'}")
                
                # Method 2: Try to find by heading if comment method failed
                if not transcript_container:
                    log_message(f"  Comment method failed, trying heading search")
                    transcript_heading = episode_soup.select_one('h2 span#transcript, h2#transcript')
                    if transcript_heading:
                        log_message(f"  Found transcript heading")
                        # Try to find the nearest containing div
                        ancestor = transcript_heading.find_parent('div')
                        if ancestor:
                            log_message(f"  Found parent div for heading")
                            transcript_container = ancestor.find('div', class_='wrap-podcast-transcript')
                            if not transcript_container:
                                # If the specific class isn't found, use the parent div itself
                                transcript_container = ancestor
                
                # Method 3: Try a direct CSS selector approach
                if not transcript_container:
                    log_message(f"  Previous methods failed, trying direct selector")
                    transcript_container = episode_soup.select_one('div.wrap-podcast-transcript')
                    if transcript_container:
                        log_message(f"  Found transcript container with direct selector")
                
                # Process transcript if found
                if transcript_container:
                    log_message(f"  Transcript container found")
                    
                    # Extract transcript paragraphs
                    transcript_paragraphs = transcript_container.find_all('p')
                    if transcript_paragraphs:
                        log_message(f"  Found {len(transcript_paragraphs)} transcript paragraphs")
                        transcript_text_list = [p.get_text(separator=' ', strip=True) for p in transcript_paragraphs]
                        transcript_text = "\n\n".join(transcript_text_list)
                        
                        # Clean up Transcript
                        cleaned_transcript_text = transcript_text.strip()
                        
                        # Save individual transcript
                        safe_title = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in episode_title)
                        filename = f"{output_folder}/{safe_title}.txt"
                        with open(filename, 'w', encoding='utf-8') as outfile:
                            outfile.write(cleaned_transcript_text)
                        log_message(f"  Transcript saved to: {filename}")
                        
                        # Append to combined transcript
                        combined_transcript_text += f"\n\n--- Episode: {episode_title} ---\n\n" + cleaned_transcript_text
                    else:
                        log_message(f"  Warning: No transcript paragraphs found within the container")
                else:
                    log_message(f"  Warning: No transcript section found")
                
                # Update progress file
                with open(progress_file, 'w') as f:
                    f.write(str(i+1))
                
            except Exception as e:
                log_message(f"  Error processing episode {episode_url}: {str(e)}")
        
        # After processing all episodes, save combined transcript
        combined_filename = f"{output_folder}/80000hours_podcast_transcripts_combined.txt"
        with open(combined_filename, 'w', encoding='utf-8') as combined_outfile:
            combined_outfile.write(combined_transcript_text.strip())
        log_message(f"\nCombined transcript saved to: {combined_filename}")
        
    except Exception as e:
        log_message(f"Error in main process: {str(e)}")
    
    log_message(f"Transcript download and combination process completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    download_80000_hours_transcripts_combined()