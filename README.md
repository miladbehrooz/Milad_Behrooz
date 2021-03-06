Welcome! This page gives an overview of my most recent and relevant Data Science projects.

# [Project 1: Image Caption Generator App](https://github.com/miladbehrooz/Image_Caption_Generator) 
----
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

# [Project 6: Classification of different objects utilizing convolutional neural networks (CNNs)](https://github.com/miladbehrooz/CNN_Object_Classifier)
Apply CNNs to classify four different objects. CNN from scratch and pertained MobileNet CNN optimized with transfer learning

![](images/object-classifier.png)
- Implemented basic algorithms in ANN (e.g. FFN and backpropagation) from scratch for educational purposes
- Generated  100 images of 4 different objects (pen, highlighter, tomato, apple) plus 160 images for 'empty' class utilizing this [python script](https://github.com/bonartm/imageclassifier)
- Built and evaluated a CNN from scratch using TensorFlow and Keras and built and evaluated a CNN based on the MobileNet CNN using the transfer learning approach
- Developed a script to real-time predict objects held into the camera

# [Project 7: Metabase Dashboard based on a PostgreSQL database using AWS (RDS/EC2)](https://github.com/miladbehrooz/PSQL_Dashboard_AWS)
In this project, I have been working with Northwind Database, a sample database shipped along with Microsoft Access (data is about 'Northwind Traders', a fictional company and its regarding sales transactions)

The project consists of three parts:

- Answered business questions on the data using PostgreSQL queries.
- Loaded the data into a PostgreSQL DB using AWS RDS and installed Metabase using AWS EC2.
- Built an interactive dashboard on a cloud server using Metabase.
