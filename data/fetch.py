import tweepy
import pandas as pd

from api_key import API_KEY, API_SECRET
auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)
api = tweepy.API(auth)


def fetch(queries: list, count, file, result_type='mixed'):
    '''Fetches @count tweets for each query in @queries and saves the results to @file.csv.
    @result_type can be either mixed, popular or recent.'''

    # filter out retweets
    queries = [q + ' -filter:retweets' for q in queries]
    tweets = []
    for query in queries:
        for tweet in tweepy.Cursor(api.search, q=query, count=50, lang='en', result_type=result_type, tweet_mode='extended').items(count):
            tweets.append([
                tweet.id_str,
                tweet.created_at,
                tweet.full_text,
                tweet.favorite_count,
                tweet.retweet_count,
                tweet.is_quote_status,
                tweet.entities,

                tweet.author.id_str,
                tweet.author.created_at,
                tweet.author.screen_name,
                tweet.author.name,
                tweet.author.description,
                tweet.author.verified,
                tweet.author.statuses_count,
                tweet.author.favourites_count,
                tweet.author.followers_count,
                tweet.author.friends_count,
                tweet.author.location,
                tweet.author.entities
            ])
    df = pd.DataFrame(tweets, columns=['tweet_id', 'tweet_date', 'tweet', 'tweet_likes', 'tweet_retweets', 'tweet_is_quote', 'tweet_entities',
                                       'user_id', 'user_date', 'user_handle', 'user_name', 'user_description', 'user_is_verified', 'user_posts',
                                       'user_likes', 'user_followers', 'user_friends', 'user_location', 'user_entities'])
    df.to_csv(f'{file}.csv', sep='\t')
    return df


if __name__ == "__main__":
    keywords = ['nigger', 'retard', 'idiot', 'bitch', 'braindead', 'fucker', 'ugly', 'nigga', 'imbecile',
                'disgusting', 'black', 'hate', 'bully', 'cancel', 'die', 'kys', 'degenerate', 'incel',
                'stupid', 'dumb', 'cunt', 'fucking', 'faggot', 'whore', 'fag', 'wanker', 'bastard',
                'twat', 'dickhead', 'hoe', 'fat', 'love', 'good', 'peace', 'cool', 'stop', 'spread',
                'cringe', 'dummy', 'lmao']
    results = fetch(keywords, 50, 'dataset')
