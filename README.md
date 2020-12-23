# How-to-Talk-Generative-Modeling-in-One-Dimension

December 23, 2020

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!

Hire me! 😊

- So we've talked a little bit about
the generative approach to classification
and today, we'll put it to use.
We'll look at a simple one-dimensional data set,
which has three classes in it.
And we'll see how we can classify
this data, by fitting a Gaussian distribution
to each of these classes.
So, here's the problem that we are dealing with.
We have a bottle of wine, and we like it,
but its label is missing.
And so we want to figure out which winery it is from.
One, two or three?
We're gonna take the machine learning approach
and we'll measure some visual and chemical features
of the bottle of wine and use this to predict
the label, one, two or three.
Now, we have our training set to help us
and this training set was obtained from 130 bottles,
43 of these bottles were from winery one,
51 from winery two, and the remaining 36
were from winery three.
For each of these bottles, a data point was measured
consisting of 13 features.
Now, these are features that makes sense
to wine experts, things like Flavanoids
and Proline and Magnesium levels.
But there are 13 said features
and so the data points are essentially
13 dimensional vectors.
So, we're gonna use this data to build a classifier
that takes the features from a new bottle
and predicts the label one, two or three.
And there's also a separate test set with 48 labeled points
that we can use to evaluate how good our classifier is.
So, let's quickly recall the generative approach,
the classification.
So, what we need to do here is to fit a distribution
to each winery individually.
So, for winery one, we fit some distribution,
say, p one of x.
For winery two, we fit a distribution, p two.
And we have a distribution, p three,
which captures the data from winery three.
We also need to know the probabilities
of each of these labels.
So, what percentage of the data,
what percentage of bottles come from winery one,
that's pi one.
What percentage of bottles come from winery two,
that's pi two and pi three.
So these pi's add up to one.
And once we have all this information,
that's our model, when we get a new bottle of wine,
x, we get its feature vector, that's the x,
and the class we predict one, two or three
is simply the class that maximizes pi j times pj of x.
So, in this case, because there are three classes,
we simply compute pi one times p one of x,
and then we compute pi two times p two of x,
and pi three times p three of x.
And we take whichever of these is the largest.
Why do we do that exactly?
Well, this is a direct consequence of Bayes' rule.
So, we have this new bottle, whose features are x.
We want to know what its label is.
And so, the label we are gonna pick
is the most likely label given x.
So, what is the probability that the label is j,
given the specific features x.
Using Bayes' rule, it's the probability
of having label j times the probability of seeing x
in label j divided by the overall probability of seeing x.
So, the probability that the label of j is just pi sub j.
The probability that we would see x
amongst the bottles from winery j is pj of x
and the denominator doesn't depend on j at all,
so we can ignore it.
So, we are simply gonna pick the label j
that maximizes this equation over here.
That's the generative approach.
So, let's go back to the data now.
This is the summary of our data set,
and the first thing we need to do
is to figure out the probabilities
of each of the individual classes.
So, out of the 130 bottles, 43 of them are from winery one.
That means the probability of winery one,
the weight of winery one is 43/130.
So, pi sub one, the weight of winery one
is 43/130, which is about 0.33.
Out of the 130 training bottles, 51 are from winery two.
So, the probability of winery two is 51/130, which is 0.39.
And likewise, the probability of winery three
is 36/130, which is 0.28.
And as you can see that these three numbers add up to one.
So, that part's easy and in general,
fitting the class weights is very easy, almost trivial.
The harder part is fitting a distribution to each class.
Now, in this case, the data consist of 13 features.
So, it's a 13-dimensional data set.
So, we need a distribution over 13 dimensional space.
We haven't yet seen how to do that.
And so, let's just simplify things for the time being
and just pick one feature out of the 13.
So, we have these 13 features,
we're just gonna pick one of them.
Let's just go ahead and choose alcohol level as our feature.
And so we've now reduced the data to just one dimension
and we're curious, how well can we predict the winery
using alcohol level alone.
Good, so now we need to fit one distribution
to the alcohol levels from winery one.
Another distribution to the alcohol levels from winery two
and a third distribution to the alcohol levels
from winery three.
What one dimensional distribution should we use?
Well, the default choice is the Gaussian.
So, let's look at it.
The Gaussian in one dimension is just
the familiar bell curve.
It's a distribution that is specified
by just two parameters, a mean and a variance.
I'm gonna be denoting the mean by the Greek letter, mu.
And the variance by the Greek letter, sigma squared.
So the mean is mu, I'm just gonna emphasize it
by rewriting it.
And the variance is sigma squared,
which means that the standard deviation is sigma.
We're gonna refer to this distribution by using
the short hand n of mu, sigma squared.
The Gaussian with mean, mu and variance sigma squared.
The n stands for normal and people call it that
because a lot of data looks like this.
Now, this particular distribution
is a distribution of real numbers.
Its density is given over here by this formula.
The density might look a little bit messy or complicated
at this point, but believe me it will start
seeming a lot more familiar as time goes on.
So, what can we say about the distribution?
So, first of all, it is centered at the mean mu,
that's right here.
It's symmetric about the mean.
About 2/3 of the distribution lies within
one standard deviation of the mean.
About 95% of the distribution lies
within two standard deviations of the mean.
So, it's a distribution that's fairly well concentrated
about its mean, and these are the distributions
that we're gonna be using.
So, we need to fit a Gaussian to the data
from each of the wineries.
Let's start from winery number one.
So we are using just one feature, alcohol.
And if you remember, we had 43 bottles from this winery.
So, we have 43 numbers, 43 alcohol levels.
And what I'm showing over here
is just a histogram plot of these numbers.
As you can see it's very jagged.
As a histogram always is, if you have very few data points.
So how do we fit a Gaussian to this?
Well, to fit a Gaussian, we just need to figure out
the mean and the variance.
So, we take our 43 numbers, the 43 alcohol levels
and we compute their mean and we compute the variance,
that's very quick, it's just two lines of code.
And what do we get?
This is the distribution we get.
The mean turns out to be 13.72.
The standard deviation is 0.44.
So sigma is 0.44, which means the variance
is the square of that, sigma squared is roughly 0.2.
And this is what the distribution looks like.
This red line is the density that we fit
for winery number one.
As you can see, it's a nice, smooth distribution
compared to the lumpy mess that we started with.
So, this will be done for winery one
and we're just gonna go ahead and do exactly
the same thing for winery two and winery three.
And this is the result.
So, we already talked about winery one.
For winery two, it turned out that
I think we had 51 bottles, so 51 numbers.
And their mean was 12.3 and the variance was 0.28.
And for winery three, the mean was 13.2
and the variance was 0.27.
And you can see the three curves over here.
These are p one, p two, and p three.
The distributions for each of the three classes.
Now, are these distributions nicely separated
from each other?
No, not really.
In fact, they're kind of on top of each other.
And this really gives us a hint that using
this one feature alone, we are not gonna be able to
classify wineries very accurately.
But let's go ahead and see what we get anyway.
So, given all these information,
given this stuff and given these pi's
and these p one, p two, p three,
how do we classify a new bottle of wine?
So, we get a new bottle of wine,
we measure its alcohol level, x,
and then, we pick the answer j, one, two or three
for which pi j times pj of x is the highest.
For example, let's say that our new bottle of wine
has an alcohol level of 15, so it's over here.
Which one are we gonna pick?
Well, the black and green ones,
that's wineries' two and three,
have almost zero density at that point.
So, we're gonna pick the red one, winery number one.
What if the bottle of wine has an alcohol level of 11?
Then we would pick winery number two, the black one.
What if it has an alcohol level of something like 12.5,
over here?
What will we pick?
Well, we wouldn't pick the red one.
We'd pick either the black or the green.
So, this is a case, in which winery two and winery three,
those two distributions, p two and p three,
have roughly the same density.
So, which one would we pick?
Well, we would pick winery number two
because it has the higher class weight.
So, remember, we are multiplying these two numbers together.
So, in that case, if the alcohol level
is something like 12.7, we would end up
picking winery number two.
So, we do all this and what kind of performance do we get?
It turns out that the test errors,
we had a test set of 48 points.
This misclassifies 14 of them and so the test error is 29%.
That's not very good, but it's way better than random.
Well, so we have seen our first classifier
using the generative approach.
It's performance was somewhat mediocre
because it was based on a single feature.
When we throw in a lot more features,
we'll be able to improve its performance dramatically.
But in order to see how to be able to do this,
we are going to delve a little further into probability.

I included some posts for reference.

https://github.com/noey2020/How-to-Talk-Probability-Review-1

https://github.com/noey2020/How-to-Talk-Generative-Approach-to-Classification

https://github.com/noey2020/How-to-Talk-of-Fitting-a-Distribution-to-Data-

https://github.com/noey2020/How-to-Talk-of-Host-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-of-Useful-Distance-Functions

https://github.com/noey2020/How-to-Talk-of-Improving-Nearest-Neighbor

https://github.com/noey2020/How-to-Talk-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-Matlab-Tricks-and-Tweaks

https://github.com/noey2020/How-to-Talk-Trading-and-Investing

https://github.com/noey2020/How-to-Work-in-Matlab-Development-Environment

https://github.com/noey2020/How-to-Talk-Vaccines

https://github.com/noey2020/How-to-Talk-Regression-in-Matlab

https://github.com/noey2020/How-to-Get-Started-in-Matlab

https://github.com/noey2020/How-to-Convert-Data-from-Web-Service-Using-Matlab

https://github.com/noey2020/Quote-for-the-Day

https://github.com/noey2020/How-to-Talk-Good-Investment-Strategy

https://github.com/noey2020/How-to-Talk-of-Good-Plan

https://github.com/noey2020/Thought-for-the-Day

https://github.com/noey2020/How-to-Talk-Stock-Watch-of-the-Day

https://github.com/noey2020/How-to-Talk-Data-Science

https://github.com/noey2020/How-to-Talk-Fundamental-Analysis

https://github.com/noey2020/How-to-Read-Company-Profiles

https://github.com/noey2020/How-to-Import-Data-from-Spreadsheets-and-Text-Files-Matlab-Without-Coding

https://github.com/noey2020/How-to-Talk-Model-of-Stock-Market-Prices-

https://github.com/noey2020/How-to-Talk-Digital-Wallets

https://github.com/noey2020/How-to-Talk-Investing

https://github.com/noey2020/How-to-Double-Your-Money-in-5years

https://github.com/noey2020/How-to-Talk-Matlab

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!
