# Aging-aware-training

First, an Aging models should be developed based on measurements of printed resistors over time (see Aging model).
Then, the training should be adapted to minimize average loss over the lifetime.

```latex
\min_\theta^{init} \quad \int_{t=0}^{t=1} L(\theta(t)) dt 
```
where the aging model describes ``` \theta(t) ``` through an equation
```latex
\theata(t) = \theta^{init} \cdot \operatorname{relativeAgingModel}(t)
```

# Evaluation

For the evaluation plots should be created displaying the accuracy(t) (y-axis) vs the time t (x-axis), for a given dataset/network.
Maybe add standarddeviations as well. 
Ideally, it can be seen that while the nonaging-aware training produceds a good initial results, the accuracy(t) degrades rather quickly, while for the networks trained with aging-aware training, the degradation is less severe and the average accuracy over the lifetime is higher.
