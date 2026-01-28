# Plans

## Experiment 1 (do-over)
This is the new description of experiment 1 that I want Claude Code to implement.

We are going to sample 20 random coordinates with mean 0 and stdev 100.

Pretraining documents will look like this (tokens in <xxs>, ... indicates omitted for bevity):
```
<start>
<point 5><point 10><12>
<point 2><point 7><13>
...
<point 6><point 5><12>
<end>
```

The pretraining documents show the distance between pairs of 20 points in arbitrary order. The coordinates are rounded to the nearest integer. Since distance is symmetric we never show the same pair of points twice (in reversed order). However the order of the pairs (both the order they occur in the document and whether it's given as <point x><point y> or <point y><point x>) is random.

For the eval, we prompt the model with a random 180 of the pairs, and measure its accuracy (MAE and pearson correlation) at reconstructing the remaining distances. It doesn't matter what order the model chooses to give those values; what matters is whether it is correct.

I want to use a non-pretrained small LLM for this. Let's try llama 3.2 1B for this unless you have a strong reason to suggest an alternative.

For the eval, log specific examples (to terminal and wandb) showing the full inputs and outputs. Do not truncate it. Also log the MAE and pearson correlations to wandb.