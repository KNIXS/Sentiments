import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\user\Downloads\sentimentdataset.csv")

def reading_csv():
    
    print(df)
    print(df.head(5))
    print(df.tail(3))
    df.info()
    print(df.describe())
    print(df.ndim)
    print(df.size)
    print(df.dtypes)
    print(df.shape)
    print(df.index)
    print(df.columns)
    print("*"*50)
reading_csv()

def data_cleaning():
    
    global df
    
    print(df.isna().sum().sum())
    df = df.dropna()
    print((df == 0).sum())
    df = df[(df != 0).all(axis=1)]
    print("*"*50)
    print((df == 0).sum())
    print(df.duplicated().sum())
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.lower()
    print(df.columns.tolist())
    df["text"] = df["text"].str.strip().replace(r"\s+", " ", regex=True)
    print("*"*50)
    
    #outliers
    
    Q_likes_1 = df["likes"].quantile(0.25)
    Q_likes_3 = df["likes"].quantile(0.75)
    
    Q_tweet_1 = df["retweets"].quantile(0.25)
    Q_tweet_3 = df["retweets"].quantile(0.75)
    
    IOR_likes = Q_likes_3 - Q_likes_1
    IOR_tweet = Q_tweet_3 - Q_tweet_1
    
    lower_likes = Q_likes_1 - 1.5*IOR_likes
    upper_likes = Q_likes_3 + 1.5*IOR_likes
    
    lower_tweet = Q_tweet_1 - 1.5*IOR_tweet
    upper_tweet = Q_tweet_3 + 1.5*IOR_tweet
    
    outliers_likes = df[(df["likes"]<lower_likes) | (df["likes"] >upper_likes) ]
    print(outliers_likes["likes"])
    
    outliers_tweet = df[(df["retweets"]<lower_tweet) | (df["retweets"] >upper_tweet) ]
    print(outliers_tweet["retweets"])
    
    print(outliers_tweet.groupby("unnamed: 0.1")[["likes","retweets"]].sum())
    
    for x in outliers_tweet.index:
        print("*"*60)
        print(df.loc[x])
        
    plt.figure(figsize=(10, 6))
        
    sns.regplot(
        data=df,
        x="likes",
        y="retweets"
        )
        
    sns.scatterplot(
        data=outliers_tweet,
        x="likes",
        y="retweets",
        color="red",
        s=100,
        label="Retweet Outliers"
        )
        
    plt.title("Likes vs Retweets — Outliers Highlighted")
    plt.xlabel("Likes")
    plt.ylabel("Retweets")
    plt.legend()
    plt.show()
    
    
data_cleaning()

def filtering():
    
    
    print("*"*60)
    print(f"Number of platforms: {df['platform'].nunique()}")
    print("Platform names:")
    print(df["platform"].unique())
    print("Most used platform is:")
    pie = df["platform"].value_counts().reset_index()
    
    pie.columns = ["platform", "users"]
    print(pie)
    print(pie.dtypes)
    
    pie["platform"] = pie["platform"].str.strip()
    pie["users"] = pd.to_numeric(pie["users"])
    
    explode = [0.1] + [0] * (len(pie) - 1)
    plt.figure(figsize=(8, 6))
    
    plt.pie(pie["users"],labels=pie["platform"],explode=explode,autopct="%1.2f%%",startangle=90)
    plt.axis("equal")
    plt.title("Platform Usage")
    plt.show()
    
    sns.violinplot(data = pie, x = "platform", y = "users", color = "red")
    plt.xticks(rotation = 45)
    plt.xlabel("platform")
    plt.ylabel("users")
    plt.show()
    
    sns.boxplot(data = pie, x = "platform", y = "users", color = "red")
    plt.xticks(rotation = 45)
    plt.xlabel("platform")
    plt.ylabel("users")
    plt.show()
    
    country = df["country"].value_counts().reset_index()
    country.columns = ["country","users"]
    
    print(country)
    print(country.dtypes)
    
    country["country"] = country["country"].str.strip()
    country["users"] = pd.to_numeric(country["users"])
    
    explode = [0.1] + [0]*(len(country)-1)
    
    sns.boxplot(data = country,x="country", y="users")
    plt.xlabel("country")
    plt.ylabel("users")
    plt.title("Country and users")
    plt.xticks(rotation = 90)
    plt.show()
    
    corr = df[["likes","retweets"]].corr()
    print(f"correlation between likes and retweets {corr}")
    print()
    print("%"*50)
    sns.heatmap(corr, annot = True, fmt=".4f" ,cmap = "coolwarm")
    plt.title("correlation between likes and retweets")
    plt.show()
    
    sentiment = df["sentiment"].value_counts().head(15).reset_index()
    sentiment.columns = ["sentiment","users"]
    sentiment["sentiment"] = sentiment["sentiment"].str.strip()
    sentiment["users"] = pd.to_numeric(sentiment["users"])
    print(sentiment)
    plt.figure(figsize=(10, 12))
    sns.barplot(data = sentiment, x = "sentiment", y = "users")
    plt.title("Number of users showed sentiments")
    plt.xlabel("sentiment")
    plt.ylabel("users")
    plt.xticks(rotation = 45,ha="right")
    plt.tight_layout()
    plt.show()
    
    sentiment_engagement = (df.groupby("sentiment")[["likes", "retweets"]].mean().sort_values("likes", ascending=False).head(15))
    print(sentiment_engagement)
    print()
    print("correlation of sentiment between likes and retweets")
    print(sentiment_engagement.corr())
    senti_corr = sentiment_engagement.corr()
    sns.heatmap(senti_corr, annot = True, fmt = ".2f", cmap = "Blues_r")
    plt.title("correlation of average of sentimet betweeen likes and tweets")
    plt.show()
    
    plt.figure(figsize=(10, 7))
    
    sns.barplot(data=sentiment_engagement.reset_index(),x="likes",y="sentiment")
    plt.title("Average Likes by Sentiment")
    plt.xlabel("Average Likes")
    plt.ylabel("Sentiment")
    plt.tight_layout()
    plt.show()


filtering()