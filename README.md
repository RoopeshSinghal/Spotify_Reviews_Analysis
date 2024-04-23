# Spotify_Reviews_Analysis
<h1>Spotify Google Play Store Review Analysis</h1>
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/d9188a23-fdf0-4d79-a7d4-9928cd5b4d08">
<h1>Contents📖</h1>
<ul>
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#datascrapingandsentimentanalysis">Data Scraping and Sentiment Analysis</a></li>
  <li><a href="#datacleaningandexploration">Data Cleaning and Exploration</a></li>
  <li><a href="#dataanalysis">Data Analysis</a></li>
  <li><a href="#findings">Conclusion</a></li>
</ul>

<h1><a name="introduction">Introduction</a></h1>
<p>In today's digital age, music streaming platforms like Spotify have become an integral part of our daily lives, offering users access to millions of songs at their fingertips. The success of such platforms hinges not only on their vast music libraries but also on the satisfaction and feedback of their users. This project aims to analyze user reviews of Spotify on the Google Play Store, utilizing advanced analytics techniques to extract valuable insights. Through sentiment analysis and thematic categorization, we'll identify areas for improvement and inform strategic decisions to enhance the user experience. Join us in uncovering the voices of Spotify users to drive continuous improvement and innovation within the platform..</p>

<ul>Tools Used🛠️:<br>
<li>Database:PostgreSQL</li>
<li>Programming Language: Python<br></li>
<li>Libraries: Pandas, Numpy, tensorflow<br></li>
<li>IDE: Vs Code,Postresql<br></li></ul>


---------------------------------------------------------------------------------------------------------------------------------
<h1><a name="datascrapingandsentimentanalysis">Data Scraping & Sentiment Analysis📊</a></h1>

<ul>Tools Used🛠️:<br>
<li>Programming Language: Python<br></li>
<li>Libraries: Pandas, Numpy, Tensorflow<br></li>
<li>IDE: Vs Code<br></li></ul>
<ul>
 <li>Import Required Libraries</li> <li>Import Required Libraries</li>
  
```python
from google_play_scraper import app, Sort ,reviews_all
from app_store_scraper import AppStore
import pandas as pd 
import numpy as np 
import json,os,uuid
```
<li>Importing Google Play Store Reviews</li>

```
sp_reviews=reviews_all(
    'com.spotify.music',
    sleep_milliseconds=0,
    lang='en',
    country='us',
    sort=Sort.NEWEST,
)
```
<li>Transforming Data Into Dataframe and Dropping columns for Analysis</li>

```
p_df=pd.DataFrame(np.array(sp_reviews),columns=['review'])
sp_df=p_df.join(pd.DataFrame(p_df.pop('review').tolist()))
sp_df.rename(columns={'reviewId':'ID','userName':'Username','content':'Review','score':'AppRating','thumbsUpCount':'ThumbsUpCount','at':'ReviewTime','replyContent':'CompanyReply','repliedAt':'ReplyTime','appVersion':'AppVersion'},inplace=True)
sp_df.drop(columns={'userImage','reviewCreatedVersion'},inplace=True)
sp_df
```

<img width="900" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/8542c4c0-9249-40f7-a12a-0151362ad233"><br>
<li>Calculating the Count of users by App Version of Spotify App</li>

```
sp_df['AppVersion'].value_counts()
```
<img width="200" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/17751586-afe6-499a-886d-55bfe71544c4"><br>
<li>Installing TensorFlow</li>

```
!pip install tensorflow
```
```
import tensorflow as tf
```
```
!pip install ipykernel
```
```
import tensorflow as tf
print(tf.__version__)
```
<img width="200" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/8265b6df-2d04-4e65-906d-612f5353f1d0"><br>
<li>Importing and Configuring Sentiment Analysis Pipeline</li>

```
from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
```
<li>Checking if sentiment analysis is working</li>

```
print(sentiment_analysis("I am working"))
```
<img width="200" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/b4dc640d-e816-4f29-9ae6-d2525e32d481"><br>
<li>Checking Data Types of DataFrame Columns</li>
```
sp_df.dtypes
```
<img width="300" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/2c1f5b0a-272f-42e5-ba69-dbdbc6c68b3c"><br>
<li>Changing Review column dtype to str for performing sentiment analysis</li>

```
sp_df['Review']=sp_df['Review'].astype('str')
```
<li>Applying Sentiment Analysis on Review Column</li>

```
sp_df['result']=sp_df['Review'].apply(lambda x: sentiment_analysis(x))
```
```
sp_df.head()
```
<img width="900" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/9cf8d94d-6903-497a-b974-d59b1e5ae1b3"><br>
<li>Extracting Sentiment Label and Score from Result Column</li>

```
sp_df['sentiment']=sp_df['result'].apply(lambda x: (x[0]['label']))
sp_df['score']=sp_df['result'].apply(lambda x: (x[0]['score']))
```
```
sp_df.head()
```
<img width="900" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/00edac13-a545-4f8d-b753-ea10bc842b0b"><br>
<li>Calculating Sentiment Distribution</li>

```
sp_df['sentiment'].value_counts()
```
<img width="300" alt="Coding" src="![image](https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/ab0f5164-01f0-4b7b-82f4-a8f4ce9d5c85)
"><br>
<li>Visualizing Sentiment Distribution</li>

```
import plotly.express as px
fig=px.histogram(sp_df,x='sentiment',color='sentiment', text_auto=True)
fig.show()
```

<img width="900" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/e2e321fd-92c1-44cb-9567-b76c13206de3"><br>
<li>Exporting DataFrame to CSV File and Reading it Back</li>

```
sp_df.to_csv("C:\\Users\\roope\\Downloads\\spotify_review.csv")
sp1_df=pd.read_csv("C:\\Users\\roope\\Downloads\\spotify_review.csv")
sp1_df
```
------------------------------------------------------------------------------------------------------------

<h1><a name="datacleaningandexploration">Data Cleaning and Exploration🧹</a></h1>

<ul><li>Tools Used🛠️:Microsoft Excel</li></ul>
<ul>
<li>Dataset</li>
<img width="200" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/7b4379f6-c76a-4a72-8516-8dd92c30a0aa"> 

<li>Deleted unwanted columns</li>
<img width="200" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/4bf41578-d00c-4cc6-91c0-9447e2c1fbc0">

<li>Checked for duplicates</li>
<img width="200" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/b2b1350e-439d-4f4f-be08-05d28bff31af">

<li>Deleted AppVersion column as there was no suitable datatype</li>
<img width="200" alt="Coding" src="![image](https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/211ced4c-4a64-4161-a368-935eb7283acc">

<li>Handling the missing values</li></ul>

<ol><li>Replaced the blank space with NULL for the column Company Reply</li>
<li>Replaced the blank space in the reply date column with a default date value which is 10-04-2024</li></ol>
<img width="200" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/453b2ab8-0c7f-4729-81a0-50be0039b1e5">

------------------------------------------------------------------------------------------------

<h1><a name="dataanalysis">Data Analysis📈</a></h1>
<ul><li>Tools Used⚙️:PostgreSQL</li></ul>

<li>Creating a schema</li>

```sql
create schema sp;

```
<li>Creating and importing dataset to postgreSQL</li>

```sql
CREATE TABLE SF1 (
  ID VARCHAR(50),
  Username VARCHAR(50),
  Review VARCHAR(5000),
  AppRating INT,
  ThumbsUpCount INT,
  ReviewTime DATE,
  CompanyReply VARCHAR(5000),
  ReplyTime DATE,
  sentiment VARCHAR(10),
  score NUMERIC );

```
```sql
COPY sr (ID,Username,Review,AppRating,ThumbsUpCount,ReviewTime,CompanyReply,ReplyTime,sentiment,score) 
FROM 'c:\Users\roope\Downloads\spotify_review.csv'
DELIMITER ','
CSV HEADER;

```

<li>Selecting and viewing the dataset</li>

```sql

select * from sr;

```
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/65959fa5-5a95-4c65-8f93-7d0df089bf10">
<li>Frequently used words in a review</li>

```sql
SELECT word, COUNT(*) AS word_count
FROM (
    SELECT regexp_split_to_table(Review, '\s+') AS word
    FROM sr
) AS words
WHERE word != ''
GROUP BY word
ORDER BY word_count DESC
limit 20;

```
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/e5db0e93-7a68-4752-8559-3e1963ff5e4c">
<li>Total No Of Reviews</li>

```sql
Select Count(*) as Total_Reviews
From sr;

```
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/0324b1af-b8cd-47db-96d0-cd589f5d4195">
<li>Longest Review</li>

```sql
SELECT Username,Review
FROM sr
ORDER BY LENGTH(Review) DESC
LIMIT 1;

```
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/aaac3737-c090-466a-9eb5-c140f84bed5c">
<li>Shortest Review</li>

```sql
SELECT Username,Review
FROM sr
ORDER BY LENGTH(Review)
LIMIT 1;

```

<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/81c49832-4109-4afc-8208-dfc1ea5cf5bb">
<li>Average Score for positive and negative reviews</li>

```sql
SELECT sentiment, AVG(score) AS avg_score
FROM sr
GROUP BY sentiment;


```
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/bb5c351e-ffa3-4dc7-8f58-66bcfae7762f">
<li>Users who are not replied by the company</li>

```sql
SELECT COUNT(*)AS not_replied
FROM sr
WHERE companyreply='NULL';;

```
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/7bc205c2-c401-422f-8735-6892eb594234">
<li>Review including word good</li>

```sql
SELECT id,review
FROM sr
WHERE Review like '%good%’;

```
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/f6603f8f-26f6-4eda-8225-0fd7ea0d9c1f">
<li>Users who are replied by the company</li>

```sql
-- Users who are replied by the company
SELECT COUNT(*)AS replied
FROM sr
WHERE companyreply !='NULL’;

```
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/f8c2c6b0-e58b-4220-8489-2f7baf5691d2">
<li>Avg Thumbsupcount by sentiment</li>

```sql
--Avg Thumbsupcount by sentiment
SELECT sentiment, AVG(ThumbsUpCount) AS avg_thumbs_up_count
FROM sr
GROUP BY sentiment;

```
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/c5414d13-00d8-4d55-be5e-20cceb22efad">
<li>Avg APP Rating</li>

```sql
--Avg APP Rating
Select AVG(AppRating) as Avg_rating
from sr;

```
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/d8006008-1593-42c0-8ef5-c5c5cf0e0422">
<li>User by most no of reviews</li>

```sql
--User by most no of reviews
SELECT Username, COUNT(*) AS num_reviews
FROM sr
GROUP BY Username
ORDER BY num_reviews DESC
LIMIT 1;

```
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/14d1371e-9b14-470a-bb65-67dc221c1e8a">
<li>Different AppRating count</li>

```sql
--Different AppRating count
SELECT AppRating, COUNT(*) AS rating_count
FROM sr
GROUP BY AppRating
ORDER BY rating_count DESC;

```
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/2bd5d8ad-3061-4aec-8e31-b0c4181db648">
<li>Most Thumbsupcount On A user's review</li>

```sql
SELECT UserName,Review, ThumbsUpCount
FROM sr
ORDER BY ThumbsUpCount DESC
LIMIT 1;

```
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/2fcaae2a-fb5e-4084-96cb-a183e795cc24">
<li>Avg Length of Review By Sentiment</li>

```sql
SELECT sentiment,AVG(LENGTH(Review)) AS avg_review_length
FROM sr
GROUP BY sentiment;

```
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/e1d9f8c4-8ce7-437a-903a-80f83563af86">
<li>Percentage of different sentiments</li>

```sql
SELECT sentiment,COUNT(*) * 100.0 / (SELECT COUNT(*) FROM sr) AS percentage
FROM sr
GROUP BY sentiment;

```
<img width="400" alt="Coding" src="https://github.com/RoopeshSinghal/Spotify_Reviews_Analysis/assets/130821105/a921d87e-fe53-481a-96f1-b374ffd10b04">
</ul>

---------------------------------------------------------------------------------------------------------
<h1><a name="findings">Conclusion</a></h1>
<p>Based on the analysis conducted in the project, the following concluions can be derived:</p>
<ol>
<li><b>Understanding User Sentiment: </b> Through an in-depth analysis of Spotify's Google Play Store reviews, we gained comprehensive insights into user sentiments regarding the platform. This holistic understanding enables us to tailor our strategies to better meet user expectations and enhance overall satisfaction.</li>

<li><b>Identifying Positive Trends:</b> The project uncovered prevalent themes and aspects consistently praised in positive reviews. These insights serve as valuable guidance for emphasizing and further improving features that resonate well with users, ultimately bolstering user satisfaction and retention.</li>

<li><b>Enhancing Product Perception:</b> By promptly addressing both positive and negative reviews, we can actively manage the perception of Spotify on the Google Play Store. Engaging with users' feedback showcases the company's commitment to customer-centricity and fosters a positive brand image.</li>

<li><b>Leveraging User Feedback for Innovation:</b> Analysis of user feedback serves as a rich source of inspiration for innovation and product enhancement. By closely examining suggestions and requests, Spotify can identify opportunities to introduce new features or refine existing ones, thereby staying ahead of evolving user needs and preferences.</li>

<li><b>Proactive Customer Support:</b> Responding to user reviews in a timely and empathetic manner demonstrates Spotify's dedication to customer support. This proactive approach not only resolves individual concerns but also instills confidence in potential users, encouraging greater adoption and loyalty.</li>

<li><b>Strengthening Community Engagement:</b> Engaging with users through review responses fosters a sense of community and belonging among Spotify users on the Google Play Store. This interaction encourages users to share their experiences and recommendations, contributing to a vibrant ecosystem around the platform.</li>

<li><b>Amplifying Positive Experiences:</b> By amplifying positive experiences shared by users, Spotify can leverage user-generated content to drive brand advocacy and organic growth. Highlighting success stories and testimonials reinforces the value proposition of the platform, attracting new users and retaining existing ones.</li>

<li><b>Continuous Improvement Culture:</b> Embracing a culture of continuous improvement based on user feedback ensures that Spotify remains responsive to evolving user preferences and market dynamics. Regular analysis and adaptation enable the platform to evolve iteratively, maintaining its relevance and competitiveness in the dynamic digital landscape.</li></ol>




