# FOLLOWUP (at most one) ? relative set-NN

After GTF (absolute min_D) over-swapped (~18) and missed W.

**One locked change:** rank / accept by **relative** set-NN distance:

```
rel(r,l) = min_D(l, S\{r}) - min_D(r, S\{r})
```

Accept only if motion PASS **and** `rel > 0` (insert farther from remaining set than removed native).
Among those, max `(rel, min_D(l), -r, -l)`.

Same pool / action space / no GT / no HGB. Still ?20 swaps.
