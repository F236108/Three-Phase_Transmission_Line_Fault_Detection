# Task 6 – Analysis

## 6.1 Logistic regression (Set R) vs Linear SVM
Logistic regression on Set R and the linear SVM score almost the same because both
are linear classifiers — they find the best separating hyperplane in the same
6-dimensional standardised feature space. Since logistic regression minimises the
log-likelihood and linear SVM maximises the margin, on a dataset that is not very
linearly separable both converge to a similarly suboptimal boundary. This tells us
that the data is **not linearly separable** in the raw feature space: a straight
decision boundary simply cannot distinguish faults from normal operation when only
looking at raw instantaneous currents and voltages.

## 6.2 Weights in Set Q
The weights on the squared current terms (Ia², Ib², Ic²) are **positive**, meaning
higher squared current values push the prediction toward fault = 1. This makes
physical sense: during a fault, the short-circuit causes current magnitudes to surge.
The weights on the squared voltage terms (Va², Vb², Vc²) are **negative**, meaning
higher squared voltage pushes toward no-fault. This also makes physical sense: during
a fault, the line voltage sags (drops toward zero) because the impedance collapses.
So faults are characterised by *high currents and low voltages*, which is exactly
what the squared-feature weights capture.

## 6.3 Support vectors: Linear vs RBF
The best linear SVM has a very large number of support vectors, close to the training
set size. This indicates that many training points are either on the wrong side of
the hyperplane or sitting right on the margin — the data is not linearly separable,
so most points are margin violators. The best RBF SVM has significantly fewer support
vectors because the RBF kernel maps the data into a higher-dimensional space where
it becomes more separable, so fewer points need to define the boundary.

## 6.4 Effect of C on accuracy and support vectors
As C increases, validation accuracy generally improves (up to a point) and the number
of support vectors tends to decrease. C controls the trade-off between margin width
and misclassification penalty: a small C allows a wider margin with more violations
(more support vectors), while a large C penalises violations heavily, yielding a
tighter fit with fewer support vectors but risking overfitting.

## 6.5 Optimistic or pessimistic test accuracy?
The test accuracy is likely **optimistic** for a real relay. Because the rows are
consecutive time samples and the split is random, neighbouring samples (which are
nearly identical in value) can end up in both training and test sets. This "data
leakage" of temporally correlated samples means the model is effectively being tested
on points very similar to its training data, inflating the reported accuracy beyond
what would be seen on genuinely unseen fault events.

## 6.6 Which model to deploy?
For a real protective relay, **missed faults** are the most dangerous error — a missed
fault means the breaker does not trip, risking equipment damage or a fire. I would
deploy the **RBF SVM** (or the logistic regression on Set Q), whichever has the fewest
missed faults in the test evaluation. Even if it has a slightly higher false-alarm rate,
false alarms only trip the breaker unnecessarily (a nuisance), while a missed fault is
catastrophic. Among the four models, the RBF SVM achieves the highest accuracy and
lowest missed-fault count, making it the safest choice for deployment.
