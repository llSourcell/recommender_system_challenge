# recommender_system_challenge
Recommender System Challenge by @Sirajology on [Youtube](https://youtu.be/9gBC9R-msAk).

##Overview

This is the code for the Recommender System challenge for 'Learn Python for Data Science #3' by @Sirajology on [YouTube](https://youtu.be/9gBC9R-msAk). The code uses the [tweepy](http://www.tweepy.org/)  library to access the Twitter API and the [TextBlob](https://textblob.readthedocs.io/en/dev/) library to perform Sentiment Analysis on each Tweet. We'll be able to see how positive or negative each tweet is about whatever topic we choose. 

The code uses the [lightfm] recommender system library to train a hybrid content-based + collaborative algorithm that uses the WARP loss function on the [movielens](http://grouplens.org/datasets/movielens/) dataset. The movielens dataset contains movies and ratings from over 1700 users. Once trained, our script prints out recommended movies for whatever users from the dataset that we choose to terminal.

##Dependencies

* numpy (http://www.numpy.org/)
* scipy (https://www.scipy.org/)
* lightfm (https://github.com/lyst/lightfm)

Install missing dependencies using [pip](https://pip.pypa.io/en/stable/installing/)

##Usage

Once you have your dependencies installed via pip, run the script in terminal via

```
python demo.py
```

**Note** If the lightfm dependency doesn't work for you via pip, just install it from source by running these two commands.

```
git clone git@github.com:lyst/lightfm.git
```
```
cd lightfm && pip install -e .
```

If you still have dependency version issues, use [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/). 

##Challenge
1. Instead of using the built-in fetch_movielens method, create your own method to fetch and parse a recommendation dataset of your choice.
You can find some good dataset options [here](https://gist.github.com/entaroadun/1653794). Make sure to look at the 
[fetch_movielens](https://github.com/lyst/lightfm/blob/master/lightfm/datasets/movielens.py#L107) method to see how it works.

2. Use 3 different loss functions (so 3 different models), compare their results, and then only print the recommendations ((products, songs, tv shows, etc.) for the best one. You'll 
find the available loss functions [here](https://github.com/lyst/lightfm/blob/master/lightfm/lightfm.py#L35).

##Credits

Credit goes to the [lightfm](https://github.com/lyst/lightfm) team. I've merely created a wrapper to make it more readable.
