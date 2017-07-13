# *2017GlobalAI2* - 2017 Global AI Hackathon San Diego Challenge 2

[Global AI Hackathon Challenge pdf](https://drive.google.com/file/d/0B2gcVmaEcT3VWno4Y2JrU1RJaGM/view)

- 3-fold cross validation on the dataset (training, (cross)-validation, and test sets) and the random shuffling of the posts (samples/examples) prior, assured the robustness of the DNN model trained upon and selected (through hyperparameter tuning); it really gives us confidence on the robustness of the accuracy of the model against other datasets to predict on.  

We had chosen to tackle Challenge 2: Make News Real Again!  It is essentially a Supervised Learning task; given labeled ("fake news" vs. "real news") data, determine some mapping or model that would then predict which class a sample/example is in (is it "fake" or "real").

## Choice of model

I learned (self-taught) Machine Learning exclusively from Prof. [Andrew Ng](http://www.andrewng.org/)'s [coursera](https://www.coursera.org/) MOOC (Massive Open Online Course) [Machine Learning Introduction](https://www.coursera.org/learn/machine-learning/home/welcome).  His direct advice in the practice of applying machine learning to a problem is to try to apply a *Deep Neural Network (DNN)* first, because in practice, you always want to have a working model first, and then iterate from there.   

# Preprocessing Facebook posts, fake or real news, without `word2vec`; using the Levenshtein distance as a metric; an explanation of `./FBfakepreprocess.ipynb`

We were given the following dataset (cf. [Global AI Hackathon Challenge pdf](https://drive.google.com/file/d/0B2gcVmaEcT3VWno4Y2JrU1RJaGM/view)) that included samples or examples of both what was labeled as "fake news" and what was "real news."  

https://github.com/BuzzFeedNews/2016-10-facebookfact-check

In the other dataset, there were all examples of fake news, with different sets of features and so it was simplest (and most direct) to first consider the Facebook posts of "real" and "fake" news with its set of features.  

Note that another team created further a larger dataset from news sources gleaned from the dataset repository out of UCI (University of California, Irvine), but it then begs the question, what is "fake news" and what is "real news" if we are not training on a single dataset that was manually labeled by one set of persons?

## Cleaning `NaN` or `NULL` values; Dealing with `NaN` values

Using Python library `pandas` for data preprocessing and file input/output with the `.csv` files, `pandas` makes it easy to deal with `NaN` values with the `.fillna` `DataFrame` method and using the `.mean` for mean (or average) value of each column.  

## Dealing with Category or author from who's writing what on FaceBook posts; from a string to a float, real number (in R).

Again, using Python library `pandas` for data preprocessing and file input/output with the `.csv` files, a look at the columns of the `.csv` file, `./data/facebook-fact-check.csv`, with the `pandas` `DataFrame` method `.columns` shows the possible features we could use to characterize the data:

```
Index([u'account_id', u'post_id', u'Category', u'Page', u'Post URL',
       u'Date Published', u'Post Type', u'Rating', u'Debate', u'share_count',
       u'reaction_count', u'comment_count', u'date_delta'],
      dtype='object')
```         

Consider `account_id` and *Category*, which told us who was posting, and what type of news source was posting ("mainstream", "left", "right", etc.).  We are given a string of numbers and a word, or string (respectively).    

In dealing with turning a string into some numerical value or numerical vector (or tensor), there is the method of `word2vec`.  I would argue that without a thoroughly thought-out and large-enough corpus that the words belong to, it would be haphazardous to apply `word2vec.`  How would even a pretrained corpus relate to your sample of words, without the specific context that the words are in?

In our case, how does one represent the column `Category`, with the arbitrarily chosen set of names?  If one considers them as classes, how does one know the total set of classes it belongs to?  The same problems go with `account_id` - how does one know in what context the account id number belongs to, if one doesn't know the set of all account ids that belong to FaceBook (which only FaceBook possibly knows)?

Thus, I chose to deploy the *[Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance)* as a metric, in relation to the *mode* (value that appears most often) of each feature (i.e. column).

## Having a *Y* value, a target value

In a supervised learning task, you need to have a *y* value, a target value, that the model predicts.  Being that the [Global AI Hackathon Challenge pdf](https://drive.google.com/file/d/0B2gcVmaEcT3VWno4Y2JrU1RJaGM/view) specified for a "model [that] should be able to determine a level of credibility", I chose to assign uniformly-divided values between 0 and 1 for each of the "classes" given in column "Rating":

* "mostly true" - 0.75
* "no factual content" - 0.5
* "mixture of true and false" - 0.25
* "mostly false" - 0.0

Then y consists of a single real value (float) between 0 and 1.

## The other numerical quantities: Dates, share count, reaction count, comment count  
It was fairly straightforward to convert dates of the given post (sample/example) into a float, or number, by choosing the earliest date posted as the origin, time zero.

The share count, reaction count, and comment count of a post were integers ranging from 0 to infinity, and so I sought to transform these features with a logarithmic function using sci-kit learn's `FunctionTransformer` in the `preprocessing` module, in the assumption that these features followed mostly a logarithmic (power-law) distribution.  

# 3-fold (training, (cross-)validation, test) cross-validation

It is **very important** to do 3-fold cross-validation split on the given dataset to make sure one doesn't overfit the model, and also (this is the reason for the (cross-) validation part of the 3-fold split) to not overfit or overlook 1 model choice over another, through the tuning of *hyperparameters.*  Indeed, this also assured that the model (finally) chosen is robust in predicting on other data sets.

I believe that the careful 3-fold cross-validation done from the start assured the robustness of our model over the other teams.

The ratio splits were 0.85 for training, 0.1 for validation, and 0.05 for testing, as a fraction of total data.

Furthermore, the posts (samples/examples) were randomly shuffled before being assigned to each fold.  This surely insured the robustness of the model that would be trained and finalized (through hyperparameter tuning).  

With data preprocessing (data wrangling/data cleaning) and cross-validation done, I looked to working out the Deep Neural Network (DNN) to train with.   

### Documents that can be edited by anyone 

DRCAT.SPACE/ffa/#12

or

[DRCAT.SPACE/ffa/#12](http://drcat.space/ffa/#12)


[github : cantren](https://github.com/cantren)



## Relevant links

[Nick Cantrell's Facebook post on wordvec](https://www.facebook.com/RoboSubNick/posts/10158742091580244)

Also found here:

goo.gl/571gSk    

and here:

goo.gl/4cEQJZ






