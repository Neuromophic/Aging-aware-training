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


