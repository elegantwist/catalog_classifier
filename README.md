Full article: http://elegantwist.com/machine-learning-for-classification/

One has to build a classifier of elements of a dictionary (company) based on a text description of the element company scope of interest.

The classification is as follows: based on the description of the element, a set of categories is chosen, to which the record relates: from the preset options, carrier type, category type, and service are selected – and consolidation us performed for these categories.

Thus, in the end, we get a catalog of type: Description — Carrier — Category — Service, and it looks like a table:

<img src="http://elegantwist.com/wp-content/uploads/2018/12/image-1.png">

where Description is set from the outside, and Carrier, Category, and Service (if not filled) are filled based on the Description

Until now, the layout — stating the categories — has been performed on a regular basis, in manual mode. As a result, the history has accumulated about the sections of the catalogs chosen, based on the given Description.

The goal has been set: automating the job – minimization, or even elimination of human involvement in the process of marking data.

The obtained results

As a result of the implementation of the solution, accuracy was achieved in 60 % of definitely correctly filled categories, with definitely incorrect filling in 5 % of cases, and 35 % of “conditionally incorrect” choice.

Further analysis of the results showed that the obtained 35 % “conditionally incorrect” classifications do not affect the overall result of the system operation (these were either new values, the classification of which was successful, but could not be verified due to the lack of history of translation, or rarely mistakenly filled records in the history), and allow to remove human involvement in the classification of the catalog with sufficient degree of confidence, by moving the classification to a fully automated process.
