[![Coverage Status](https://coveralls.io/repos/github/5amron/pre-ml/badge.svg?branch=master)](https://coveralls.io/github/5amron/pre-ml?branch=master) [![Build Status](https://travis-ci.org/5amron/pre-ml.svg?branch=master)](https://travis-ci.org/5amron/pre-ml) [![PyPI version](https://badge.fury.io/py/pre-ml.svg)](https://badge.fury.io/py/pre-ml) [![License: MIT](https://camo.githubusercontent.com/890acbdcb87868b382af9a4b1fac507b9659d9bf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d4d49542d626c75652e737667)](https://opensource.org/licenses/MIT)










# pre-ml ![](http://pre-ml.com/static/img/ant-samll.png "Logo Title Text 1")


A tool for various optimization problems such as feature selection in machine learning. Feature selection (FS) is the process of selecting a subset of relevant features (variables, predictors) for use in the model construction and it is one of most important steps which can affect the performance of a pattern recognition system. The ultimate goal of feature selection is to select a feature subset from the original feature set to increase the performance of learning algorithms. pre-ml sees the feature set as a fully connected graph and uses binary ant colony optimization in order to optimize datasets for classification problems. At the end solution will be given you by several ways, for more info checkout the [docs](http://pre-ml.com/docs/).


package website :
http://pre-ml.com/




## Installation

pre-ml is available on PyPI in fully stable version. This package is going to have more features soon, be in touch please!

```
pip install pre-ml
```



## Online Run
web app giving the capability of preprocessing on datasets to non-programmer users based on simple and easy to use graphical user interface settings. This section begins with setting the preferences and uploading the user datasets which consists of two files, first data file and second targets file. The structure of these files have been discussed in website. users after uploading and choosing their settings which each one of them has been explained in some proper way can hit the run button in order to begin the optimization of their dataset.
When processing is finished, results are displayed to users in several charts and paragraphs containing all of information about their dataset and the optimization process that has been done on it. Basically two charts will be shown, first one displaying best accuracy of dataset in each iteration, so they can find out about progress that has been made and second the number of features that are selected in each iteration.


http://pre-ml.com/run/









Saman Sadeghyan
s.sadeghy325@gmail.com
