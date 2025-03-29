import praw

# --- Replace with your actual credentials ---
client_id = "CQqDLcqLl_Mze9y9pmMpCQ"
client_secret = "hwgmVDG2jkPyxrWaYiSdoYOpOt-5aw"
user_agent = "python:EAScraper:v1.0 (by /u/UsedToBeaRaider)"
reddit_username = "UsedToBeaRaider"

# --- Initialize PRAW ---
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
    username=reddit_username  # Optional, but good practice to include
)

# --- Specify the subreddit ---
subreddit_name = "EffectiveAltruism"
subreddit = reddit.subreddit(subreddit_name)

# --- Parameters for scraping ---
limit_posts = 500  # Number of top posts to fetch (adjust as needed)
time_filter = "all"  # Time period for "top" posts (all, day, week, month, year)
comment_limit_per_post = 3 # Number of top comments per post to retrieve

# --- Lists to store data ---
posts_data = []
comments_data = []

# --- Fetch top posts ---
print(f"Fetching top {limit_posts} posts from r/{subreddit_name} (time filter: {time_filter})...")
top_posts = subreddit.top(time_filter=time_filter, limit=limit_posts)

for post in top_posts:
    post_info = {
        "post_id": post.id,
        "title": post.title,
        "url": post.url,
        "score": post.score,
        "upvote_ratio": post.upvote_ratio,
        "num_comments": post.num_comments,
        "author": str(post.author), # Convert author object to string
        "created_utc": post.created_utc,
        "selftext": post.selftext, # Post body (if it's a self-post)
        "permalink": "https://www.reddit.com" + post.permalink
    }
    posts_data.append(post_info)
    print(f"  Post: '{post.title}' (ID: {post.id}, Score: {post.score})")

    # --- Fetch top comments for each post ---
    post.comment_sort = "top" # Sort comments by "top"
    post.comments.replace_more(limit=0) # Reduce 'load more comments' chains
    top_comments = post.comments.list()[:comment_limit_per_post] # Get top N comments

    for comment in top_comments:
        comment_info = {
            "post_id": post.id,
            "comment_id": comment.id,
            "comment_author": str(comment.author), # Convert author object to string
            "comment_body": comment.body,
            "comment_score": comment.score,
            "comment_permalink": "https://www.reddit.com" + comment.permalink,
            "comment_created_utc": comment.created_utc,
            "is_submitter": comment.is_submitter # True if comment author is post author
        }
        comments_data.append(comment_info)
        print(f"    - Comment by {comment_info['comment_author']} (Score: {comment_info['comment_score']})")


print("\n--- Scraping complete! ---")
print(f"Fetched {len(posts_data)} posts and {len(comments_data)} comments.")

# --- (Optional) Save data to CSV using pandas ---
import pandas as pd

posts_df = pd.DataFrame(posts_data)
comments_df = pd.DataFrame(comments_data)

posts_csv_filename = f"ea_reddit_posts_{subreddit_name}_{time_filter}_{limit_posts}.csv"
comments_csv_filename = f"ea_reddit_comments_{subreddit_name}_{time_filter}_{limit_posts}.csv"

posts_df.to_csv(posts_csv_filename, index=False)
comments_df.to_csv(comments_csv_filename, index=False)

print(f"\nData saved to:")
print(f"- Posts: {posts_csv_filename}")
print(f"- Comments: {comments_csv_filename}")

print("\n--- Script finished ---")