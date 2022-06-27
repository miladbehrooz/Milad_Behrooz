Welcome! This page gives an overview of my most recent and relevant Data Science project.

# [Project 1: Image Caption Generator App](https://github.com/miladbehrooz/Image_Caption_Generator) 
A Deep learning-based web application for captioning images developed using CNN and LSTM models

![](images/img-cap-demo.gif)
- Built model based on CNN and LSTM to generate captions for images automatically
- Trained deep learning model on Flickr8K dataset. 
- Built an image caption generator  web application with Streamlit based on the deep learning model  

# [Project 2: Movie Recommender App](https://github.com/miladbehrooz/Movie_Recommender)
A movie recommender web application based on an unsupervised learning method to suggest movies based on user input

![](images/movie-rec-demo.gif)
- Used small dataset of [MovieLens](https://grouplens.org/datasets/movielens/) (100,000 ratings applied to 9,000 movies by 600 users) and webscrape movie posters from [OMDb API](http://www.omdbapi.com/)
- Implemented the following recommender methods:
  - Simple Recommender (recommend the most popular movies)
  - Non Negative Matrix Factorization (NMF)
  - Collaborative Filtering
- Built a movie recommender app with Flask - user can select favorites movies. When user rate selected movies, 5 movies based on NMF algorithm  are recommended 

# [Project 3: A Dockerized Data Pipeline for Sentiment Analysis on tweets](https://github.com/miladbehrooz/Dockerized_Data_Pipeline)
A data pipeline with Docker to perform Sentiment Analysis on tweets and post it on a slack channel via a bot
There are 5 steps in data pipeline:
- Extract tweets with [Tweepy API](https://docs.tweepy.org/en/stable/index.html) 
- Load the tweets in a MongoDB
- Extract the tweets from MongoDB, perform sentiment analyisis on the tweets and load the transformed data in a PostgresDB (ETL job)
- Load the tweets and corresponding sentiment assessment in a Postgres database
- Extract the data from the PostgresDB and post it in a slack channel with a slackbot

![](images/docker-workflow.jpg)

# [Project 4: Supermarket Simulation](https://github.com/miladbehrooz/Supermarket_Simulation)
Simulation of the customer behavior in a supermarket with Markov Chain Modeling and focusing on OOP

The project consists of three part:
- Performing Exploratory Data Analysis
- Calculation of the Transition Probabilities
- Implementation of a Markov Chain-based simulator (for one or multiple customers)

# [Prpject 5: Timeseries Analysis Temperature](https://github.com/miladbehrooz/Timeseries_Analysis_Temperature)
Carried out step by step time series analysis of temperature data using Trend Seasonal model , AR model, and ARIMA model

![](images/temp.gif)

- Prepared a temperature forecast using temperature data from [the website of the European Climate Assessment & Dataset project](https://www.ecad.eu/)
- Performed data cleaning and did a step-by-step time series analysis of the data, starting a base model to mimic trend and seasonality
- Built and evaluated  AR and ARIMA models.

![](images/prediction_2021.png)
Figure : Prediction of temperature for Berlin-Tempelhof station via differnet models for 2021
