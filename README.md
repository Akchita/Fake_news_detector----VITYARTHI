# Fake_news_detector----VITYARTHI
The Fake News Detector is a Machine Learning-based project built using the Python programming language, which attempts to classify the given news article as real or fake. With the increasing rate of fake news being spread online, the Fake News Detector attempts to identify the fake news content using natural language processing.



# News Detector using Machine Learning



## Project Overview

This project is a system that detects news. The News Detector is designed to figure out if a news story's fake or real.

The system uses a model to classify news text as fake or real.

The News Detector takes user input then it predicts whether the news is trustworthy or not.

This is a tool in todays world where there is a lot of fake news.

The News Detector is a system.

## Features

* The News Detector is simple to use.

* It uses a technique called TF-IDF Vectorization for text processing.

This means that it looks at the words in the news story and decides if they are important or not.

* The News Detector uses a model for classification.

This means that it takes the words and decides if the news story is fake or real.

* The News Detector has a command-line interface.

This means that you can type in a news story and it will tell you if it is fake or real.

* The code for the News Detector is easy to understand.

## Technologies Used

* The programming language used for the News Detector is Python.

* The News Detector uses Pandas to work with data.

* The News Detector uses a library called Scikit-learn for Machine Learning.

* The News Detector uses TF-IDF for text processing.

## Project Structure

```

📁 News-Detector

── news.csv           # Dataset file

── main.py            # Python script

│── README.md          # Project documentation

```

## Dataset

* The dataset for the News Detector should be a CSV file named news.csv.

* The dataset must have least two columns:

* text → this is where you put the news story

* label → this is where you say if the news story is fake or real

Here is an example:

| text                      | label |

| ------------------------- | ----- |

Government passes new law | real  |

| Aliens landed in India    | fake  |

## Installation & Setup

### 1. Install Required Libraries

To install the required libraries you need to run this command:

```bash

pip install pandas scikit-learn

```

### 2. Place Dataset

You need to make sure that news.csv is in the folder as your Python file.

### 3. Run the Program

To run the program you need to type this command:

```bash

python main.py

```

## How It Works

### 1. Load Dataset

The News Detector reads the data from news.csv.

This is where it gets all the news stories and their labels.

### 2. Text Cleaning

The News Detector converts all the text to lowercase and removes punctuation.

This helps the News Detector to understand the text

### 3. Train-Test Split

The News Detector splits the data into two parts:

* training data

* testing data

This is where it uses some of the data to train the News Detector and some of it to test the News Detector.

### 4. Feature Extraction

The News Detector converts the text into a form using TF-IDF.

This is where it looks at all the words in the news story and decides which ones are important.

### 5. Model Training

The News Detector uses a model to learn patterns from the data.

This is where it takes all the words and decides if the news story is fake or real.

### 6. Prediction

The News Detector takes user input then it predicts if the news story is fake or real.

This is where you can type in a news story and it will tell you if it is trustworthy or not.

## Usage

After you run the program you will see this:

```

===== NEWS DETECTOR =====

Enter news text (or type 'exit'):

```

Here are some examples:

```

Enter news text: Scientists discover water on Mars

✅ This news is REAL

```

```

Enter news text: Celebrity is actually an alien

❌ This news is

```

## Output

* The News Detector shows you how accurate it is.

* The News Detector gives you a prediction for the news story you typed in.

## Limitations

* The accuracy of the News Detector depends on the quality of the data.

* The News Detector cannot verify news stories in time.

* The News Detector only works on patterns it has learned from the data.

## Future Improvements

* Add a graphical user interface using Tkinter or Streamlit.

* Use models, like Bayes, Random Forest or Deep Learning.

* Add a real-time news API integration.

## Author

This project was made by Aditi.

It is a beginner AI/ML project.

## If you like this project

Give it a star. Keep learning about AI.
