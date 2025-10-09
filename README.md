# Customer Sentiment Analysis - Floweraura



As a Data Analyst at Amazon, you have been tasked with gauging customer sentiment towards the iPhone 15 128GB model. The primary goal of this project is to analyze public perception and evaluate customer reactions by performing sentiment analysis on product reviews posted by users. By extracting and processing customer reviews, you will derive insights about the overall sentiment (positive or negative) surrounding the product, which can be useful for decision-making, improving customer experience, and identifying key areas for product improvement.

## Objective
1. Scrape reviews of the Red Roses Bouquet from FlowerAura’s product pages.

2. Clean and preprocess review data for sentiment analysis.
   
4. Use TextBlob to compute sentiment polarity and classify reviews as positive or negative.
5. Visualize results through graphs and word clouds to highlight customer opinions and recurring themes.
6. Summarize findings and key recommendations for business improvement.

 ## Tools and Libraries
BeautifulSoup – Web scraping reviews

Requests – Fetching page content

Pandas – Data handling and analysis

TextBlob – Sentiment analysis

Matplotlib / Seaborn – Visualization

WordCloud – Keyword insights

## Tasks

## 1. Data Collection (Web Scraping):

**Tool:** Selenium and BeautifulSoup

**Task:** Scrape at least 300 customer reviews from Flipkart's product page for the iPhone 15 128GB model. Each review should include:

**Username:** The name of the reviewer.

**Rating:** The rating provided by the user (1 to 5 stars).

**Review Text:** The content of the customer's review, which may contain valuable information regarding their experience with the product.

**Steps:**

Set up Selenium to automate browser interactions, navigate to Flipkart’s product page for iPhone 15 128GB, and extract the reviews.
Use BeautifulSoup to parse the HTML of the reviews and extract the relevant details (username, rating, and review text).
Ensure that the scraper handles pagination to retrieve reviews from multiple pages if necessary.


# Import the necessary librariess
```python
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
```

## Create empty lists to store the user data such as Name, Ratings, Reviews, Cities, occasions, posted_on
```python
Names =[]
Ratings = []
Reviews = []
Cities = []
Occasions = []
Posted_On = []
```


## 2. Data Cleaning and Preprocessing:

**Tool:** Pandas

**Task:** Clean and preprocess the scraped data for analysis.

**Steps:**

**Remove duplicates:** Eliminate any duplicate reviews to ensure data quality.
Handle missing values: Address missing or incomplete data, such as missing review text or rating, by either removing rows or filling in missing values if applicable.

**Text preprocessing:**

Convert the review text to lowercase.
Remove irrelevant characters (e.g., special characters, punctuation, and extra spaces).
Tokenize the text into individual words.
Remove stop words (commonly used words that do not add significant meaning to sentiment analysis).
Perform lemmatization to convert words into their base form (e.g., "running" → "run").

## Assign the scraped dataset to a dataframe
```python
df = pd.DataFrame({"Names":Names, "Cities":Cities, "Posted_On":Posted_On, "Occasions":Occasions, "Ratings":Ratings, "Reviews":Reviews})
df
```
![Dashboard Screenshot](https://github.com/RushiSonar123/Customer_Sentimental_Analysis/blob/main/Data%20Frame%20after%20cleaning.png)
```python
###Extracting and Cleaning from Posted_On and Occassions Columns
def extract(value):
    try:
        x = value.index(':')
        return value[x+2:]
    except:
        return np.nan

df["Posted_On"] = df["Posted_On"].apply(extract)
df["Occasions"] = df["Occasions"].apply(extract)
df
```
![Dashboard Screenshot](https://github.com/RushiSonar123/Customer_Sentimental_Analysis/blob/main/After%20removing%20extra%20text%20from%20columns.png)
## Removing (th, rd,st,nd) from Posted_On Columns
```python
#Removing (th, rd,st,nd) from Posted_On Columns
rep = ["th", "rd", "st", "nd"]

for i in rep:
    df["Posted_On"] = df["Posted_On"].str.replace(i, "")
df
```
```python
# Checking the datatype of each.
df.info()
```
![Dashboard Screenshot](https://github.com/RushiSonar123/Customer_Sentimental_Analysis/blob/main/Checking%20the%20datatype%20of%20each..png)


## 3. Sentiment Analysis:

**Tool:** TextBlob

**Task:** Analyze the sentiment of each review to classify them as either positive or negative.

**Steps:**

Use TextBlob to perform sentiment analysis on the review text.
TextBlob will provide a polarity score between -1 (negative) and +1 (positive), as well as a subjectivity score.

Define a threshold to classify the sentiment:
Positive sentiment: Polarity score ≥ 0.1

Negative sentiment: Polarity score < 0.1

Store the sentiment classification for each review in the dataset.
```python
df["Posted_On"] = pd.to_datetime(df["Posted_On"])
df["Ratings"] = df["Ratings"].astype("float")
df["Polarity"] = [TextBlob(i).sentiment.polarity for i in df["Reviews"]]
df["Subjectivity"] = [TextBlob(i).subjectivity for i in df["Reviews"]]

# Calculates and prints the overall average polarity score of the entire dataset of reviews
p = df["Polarity"].mean()
if p <=-0.3:
    print("negative")
elif p <0.3:
    print("neutral")
else:
    print("positive")

# Function to assign the Class to the Polarity
def score(value):
    if value <= 0:
        return "negative"
    else:
        return "positive"

df["Score"] = df["Polarity"].apply(score)
df
```
![Dashboard Screenshot](https://github.com/RushiSonar123/Customer_Sentimental_Analysis/blob/main/Polarity%20Score.png)

```python
# Plotting score and there count 
ax = sns.countplot(x = df['Score'], data = df)
ax.bar_label(container=ax.containers[0])
plt.show()
```
![Dashboard Screenshot](https://github.com/RushiSonar123/Customer_Sentimental_Analysis/blob/main/Sentiment%20Distribution.png)


## Data Analysis and Insights:

**Tool:** Pandas and Matplotlib/Seaborn for visualization

**Task:** Perform an analysis on the sentiment of reviews and extract actionable insights.

**Steps:**

Sentiment Distribution: Calculate the overall distribution of positive and negative sentiments for the 300 reviews.
Average Rating vs Sentiment: Analyze if there is any correlation between the numeric ratings (1-5 stars) and sentiment polarity. Do higher ratings correspond with more positive sentiments?
Word Cloud: Create a word cloud to identify the most frequently mentioned words in the positive and negative reviews.
Review Length Analysis: Investigate if longer reviews are associated with more detailed sentiments, either positive or negative.

```python
# Positive reviews word cloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
df_pos = df.loc[(df["Score"] == "positive")]
df_neg = df.loc[(df["Score"] == "negative")]
all_text = " ".join(text for text in df['Reviews'])
wordcloud  = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

```
![Dashboard Screenshot](https://github.com/RushiSonar123/Customer_Sentimental_Analysis/blob/main/Positive%20review%20word%20cloud.png)

```python
# Negative reviews word cloud
all_text = " ".join(text for text in df_neg['Reviews'])
wordcloud  = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

![Dashboard Screenshot](https://github.com/RushiSonar123/Customer_Sentimental_Analysis/blob/main/Negative%20review%20word%20cloud.png)

```python
# Correlation between Rating and Polarity
correlation = df["Ratings"].corr(df["Polarity"])
print("Correlation between Ratings and Sentiment Polarity:", correlation)

# Scatter plot to visualize relationship
plt.figure(figsize=(8,5))
plt.scatter(df["Ratings"], df["Polarity"], alpha=0.6, color="blue")
plt.title("Ratings vs Sentiment Polarity")
plt.xlabel("Ratings (1-5 stars)")
plt.ylabel("Sentiment Polarity")
plt.axhline(0, color="red", linestyle="--")
plt.show()
```

![Dashboard Screenshot](https://github.com/RushiSonar123/Customer_Sentimental_Analysis/blob/main/Corelation.png)

```python
# Average polarity per rating
avg_polarity = df.groupby("Ratings")["Polarity"].mean()
print("\nAverage Sentiment Polarity by Rating:\n", avg_polarity)

# Plot average polarity
avg_polarity.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Average Sentiment Polarity per Rating")
plt.xlabel("Ratings")
plt.ylabel("Average Polarity")
plt.show()
```
![Dashboard Screenshot](https://github.com/RushiSonar123/Customer_Sentimental_Analysis/blob/main/Rating%20vs%20sentiment%20Polarity.png)

![Dashboard Screenshot](https://github.com/RushiSonar123/Customer_Sentimental_Analysis/blob/main/Average%20Sentiment%20Polarity%20by%20Rating.png)

```python
# Add Review Length (number of words per review)
df["Review_Length"] = df["Reviews"].apply(lambda x: len(x.split()))

# Correlation between review length and sentiment polarity
length_corr = df["Review_Length"].corr(df["Polarity"])
print("Correlation between Review Length and Sentiment Polarity:", length_corr)

# Scatter plot: Review length vs Polarity
plt.figure(figsize=(8,5))
plt.scatter(df["Review_Length"], df["Polarity"], alpha=0.5, color="green")
plt.title("Review Length vs Sentiment Polarity")
plt.xlabel("Review Length (words)")
plt.ylabel("Sentiment Polarity")
plt.axhline(0, color="red", linestyle="--")
plt.show()
```
![Dashboard Screenshot](https://github.com/RushiSonar123/Customer_Sentimental_Analysis/blob/main/Correlation%20between%20review%20length%20and%20sentiment%20polarity.png)

```python
# Compare average length between positive and negative reviews
avg_length = df.groupby("Score")["Review_Length"].mean()
print("\nAverage Review Length by Sentiment Score:\n", avg_length)

# Boxplot to visualize
import seaborn as sns
plt.figure(figsize=(8,5))
sns.boxplot(x="Score", y="Review_Length", data=df, palette="Set2")
plt.title("Review Length Distribution by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Review Length (words)")
plt.show()
```
![Dashboard Screenshot](https://github.com/RushiSonar123/Customer_Sentimental_Analysis/blob/main/Review%20Length%20vs%20sentiment%20polarity.png)

# Customer Sentiment Analysis - Floweraura
**1. Overview of Data Collection and Cleaning**

**Source:** Reviews scraped from FlowerAura’s product page (10 pages of reviews).

**Data Collected:** Reviewer names, city, occasion, posted date, ratings, and review text.

**Cleaning Steps:**

Extracted and standardized review dates (removed suffixes like st, nd, rd, th).

Converted ratings from text to numeric (1–5 stars).

Applied TextBlob to compute:

**Polarity** → Sentiment score (–1 = negative, +1 = positive).

**Subjectivity** → Degree of opinion vs fact.

**Added derived features:** Review Length (word count) and Score (positive/negative based on polarity).

**2. Sentiment Analysis Results**

*Distribution of Sentiment:*

Majority of reviews are positive, reflecting overall customer satisfaction.

A smaller portion are negative, but those reviews tend to be longer and more detailed.

Average Sentiment per Rating:

* 5★ ratings* → Strongly positive polarity (customers happy with freshness, presentation, and delivery).

* 3★ ratings* → Mixed polarity (neutral or balanced opinions).

* 1★–2★ ratings* → Strongly negative polarity (complaints about delivery, freshness).

**Correlation:**

Positive correlation between numeric rating and sentiment polarity (higher rating = more positive sentiment).

**Review Length Insights:**

*Negative reviews are longer* → unhappy customers provide detailed complaints.

*Positive reviews are shorter* → often quick praises like “Beautiful bouquet” or “Loved it”.

**3. Insights**
 **Positive Highlights:**

**Customers appreciate:**

Freshness of roses.

Beautiful presentation of the bouquet.

On-time delivery on special occasions.

Emotional satisfaction when gifting (Anniversaries, Birthdays, Valentine’s Day).

 **Common Issues:**

Late delivery reported in some cities.

Freshness not consistent (wilted flowers in a few cases).

Size/quantity mismatch (bouquet smaller than expected).

Packaging concerns (roses not arranged properly).

**4. Recommendations**

* **Product Quality:**

Ensure strict freshness checks before dispatch.

Improve packaging to maintain bouquet shape during transit.

* **Delivery Experience:**

Strengthen last-mile logistics to ensure timely delivery, especially on occasions.

Introduce delivery time-slot guarantees (customers value punctuality for gifting).

* **Customer Transparency:**

Set clear expectations about bouquet size/quantity to avoid disappointment.

Highlight freshness guarantee in product descriptions and marketing.

* **Marketing Opportunities:**

Leverage positive reviews (freshness, happiness of recipients) in ads.

Promote occasion-specific campaigns (Valentine’s Day, Anniversaries).

Offer loyalty discounts for repeat customers who gift flowers frequently.















