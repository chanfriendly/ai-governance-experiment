import requests
from bs4 import BeautifulSoup
import csv
import time  # For adding delays

def scrape_ea_forum_post(post_url):
    """Scrapes an individual EA Forum post page for main content and filtered comments."""
    try:
        response = requests.get(post_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # --- Extract Main Post Content ---
        main_post_element = soup.find('div', id='postContent') # Using #postContent based on previous analysis!
        main_post_content = main_post_element.get_text(separator='\n', strip=True) if main_post_element else "N/A"

        # --- Extract and Filter Comments ---
        comments_data = []
        # --- Updated comment container selector - still using 'comment' class, but verify ---
        comment_containers = soup.find_all('div', class_='CommentsItem-body') # Changed to CommentsItem-body, based on your snippet!

        for comment_container in comment_containers:
            # --- Updated comment selectors based on your snippet ---
            comment_author_element = comment_container.select_one('.CommentUserName-author') # Using .CommentUserName-author for author
            comment_body_element = comment_container.select_one('.CommentBody-commentStyling p') # Using .CommentBody-commentStyling p for body
            comment_score_element = comment_container.select_one('.OverallVoteAxis-voteScore') # Using .OverallVoteAxis-voteScore for score

            comment_author = comment_author_element.get_text(strip=True) if comment_author_element else "N/A"
            comment_body = comment_body_element.get_text(separator='\n', strip=True) if comment_body_element else "N/A"
            comment_score = int(comment_score_element.get_text(strip=True)) if comment_score_element and comment_score_element.get_text(strip=True).isdigit() else 0

            # --- Filter comments (adjust thresholds as needed) ---
            if comment_score >= 5: # Example: Filter for comments with score of 5 or higher
                comment_info = {
                    "comment_author": comment_author,
                    "comment_body": comment_body,
                    "comment_score": comment_score
                }
                comments_data.append(comment_info)

        return {"main_post_content": main_post_content, "filtered_comments": comments_data}

    except requests.exceptions.RequestException as e:
        print(f"Error scraping post {post_url}: {e}")
        return {"main_post_content": "Error fetching post", "filtered_comments": []}


# --- Level 1: Scrape Recommendations Page (no changes here) ---
recommendations_url = "https://forum.effectivealtruism.org/recommendations"

try:
    response = requests.get(recommendations_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    recommended_posts_data = []
    recommendation_containers = soup.find_all('div', class_='EAPostsItem-root')

    if recommendation_containers:
        print(f"Found {len(recommendation_containers)} recommended posts on recommendations page...")
        for rec_container in recommendation_containers:
            title_element = rec_container.find('span', class_='PostsTitle-eaTitleDesktopEllipsis').find('a')
            karma_element = rec_container.find('div', class_='EAKarmaDisplay-root').find('span', class_='LWTooltip-root')
            author_element = rec_container.find('a', class_='UsersNameDisplay-noColor')
            date_element = rec_container.find('span', class_='EAPostMeta-date').find('time')
            comments_element = rec_container.find('a', class_='EAPostsItem-comments')

            if title_element and 'href' in title_element.attrs:
                post_title = title_element.get_text(strip=True)
                post_relative_link = title_element['href']
                post_url = "https://forum.effectivealtruism.org" + post_relative_link if not post_relative_link.startswith('http') else post_relative_link
                post_karma = int(karma_element.get_text(strip=True)) if karma_element else 0
                post_author = author_element.get_text(strip=True) if author_element else "N/A"
                post_date = date_element['datetime'] if date_element and 'datetime' in date_element.attrs else "N/A"
                post_comments_count = int(comments_element.get_text(strip=True)) if comments_element else 0

                print(f"  - Processing recommended post: '{post_title}' - {post_url}")
                post_data = scrape_ea_forum_post(post_url) # Level 2 scraping

                recommended_posts_data.append({
                    "recommendation_page_url": recommendations_url,
                    "post_title": post_title,
                    "post_url": post_url,
                    "post_karma": post_karma,
                    "post_author": post_author,
                    "post_date": post_date,
                    "post_comments_count": post_comments_count,
                    "main_post_content": post_data["main_post_content"],
                    "filtered_comments": post_data["filtered_comments"]
                })
                time.sleep(1) # Be polite, add a delay between requests

    else:
        print("No recommendation containers found on recommendations page. HTML structure might have changed.")


    # --- Save data to CSV (no changes here) ---
    csv_filename = "ea_forum_recommendations_posts_and_comments.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["recommendation_page_url", "post_title", "post_url", "post_karma", "post_author", "post_date", "post_comments_count", "main_post_content", "comment_author", "comment_body", "comment_score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for post_item in recommended_posts_data:
            for comment in post_item["filtered_comments"]:
                writer.writerow({
                    "recommendation_page_url": post_item["recommendation_page_url"],
                    "post_title": post_item["post_title"],
                    "post_url": post_item["post_url"],
                    "post_karma": post_item["post_karma"],
                    "post_author": post_item["post_author"],
                    "post_date": post_item["post_date"],
                    "post_comments_count": post_item["post_comments_count"],
                    "main_post_content": post_item["main_post_content"],
                    "comment_author": comment["comment_author"],
                    "comment_body": comment["comment_body"],
                    "comment_score": comment["comment_score"]
                })
            if not post_item["filtered_comments"]:
                 writer.writerow({
                    "recommendation_page_url": post_item["recommendation_page_url"],
                    "post_title": post_item["post_title"],
                    "post_url": post_item["post_url"],
                    "post_karma": post_item["post_karma"],
                    "post_author": post_item["post_author"],
                    "post_date": post_item["post_date"],
                    "post_comments_count": post_item["post_comments_count"],
                    "main_post_content": post_item["main_post_content"],
                    "comment_author": "N/A",
                    "comment_body": "N/A",
                    "comment_score": "N/A"
                })


    print(f"\nData saved to CSV file: {csv_filename}")


except requests.exceptions.RequestException as e:
    print(f"Error fetching recommendations page: {e}")