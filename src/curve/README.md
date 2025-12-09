# Dynamic curve modelling

The Nelson-Siegel model (1987),
then reparameterized by Diebold \& Li in 2006,
is a famous macro model for the term structure of yields.

It embeds that most of the information contained in the term structure
are :
- the long-term value of yields (which will eventually flatten),
- the short term slope,
- a smooth dynamic at medium range (belly of the curve).

Given a set of observations $y_i(t_j)$ of yields or zero-volatility spreads,
for issuer $i$ and time indexed $j$,
one can use OLS to fit the NS model.

For issuers with only a few points (quotes),
OLS is not robust :
- interpolation is not necessarily arbitrage-free in practice,
- the level can become negative,
- excessively wide variations can occur, without fundamental justification.

To cope with scarcely-quoted issuers, one can apply 
optimisation under constraints (using the Lagrangian),
or use a probabilitic approach with a well suited prior 
for the state-space parameters 
(for instance, the prior for the level could follow an exponential or gamma law).
The latter solution (which is a MLE) enables one to incorporate historical knowlege, 
or market expectations.

Then, moving forward in time, the parameters can be dynamically updated :

```math
\begin{cases}
X_{t+1} = A_t X_t + B_t b(t) + \eta_t \\
Y_{t+1} = H_t X_{t+1} + \epsilon_{t+1}
\end{cases}
```

Where $A_t$ would be derived from the gradient of the NS state-space,
depending on the method of optimisation for a given issuer.

For instance, for OLS, $A$ would describe how to adapt the current
optimal solution to the next (or rather at the first order) optimal solution 
(at $t+1). 


## Roadmap
1. flat curve model: one parameter per issuer, the level,
2. alignment with data internal structures, 
3. OLS, then with dynamic updates,
4. other optimisation methods: Lagragian and Bayesian models.

## References

```bibtex
@article{coroneo2011arbitrage,
  title={How arbitrage-free is the Nelson--Siegel model?},
  author={Coroneo, Laura and Nyholm, Ken and Vidova-Koleva, Rositsa},
  journal={Journal of Empirical Finance},
  volume={18},
  number={3},
  pages={393--407},
  year={2011},
  publisher={Elsevier}
}
```

```bibtex
@article{diebold2008global,
  title={Global yield curve dynamics and interactions: a dynamic Nelson--Siegel approach},
  author={Diebold, Francis X and Li, Canlin and Yue, Vivian Z},
  journal={Journal of Econometrics},
  volume={146},
  number={2},
  pages={351--363},
  year={2008},
  publisher={Elsevier}
}
```

```bibtex
@article{diebold2005modeling,
  title={Modeling bond yields in finance and macroeconomics},
  author={Diebold, Francis X and Piazzesi, Monika and Rudebusch, Glenn D},
  journal={American Economic Review},
  volume={95},
  number={2},
  pages={415--420},
  year={2005},
  publisher={American Economic Association}
}
```

```bibtex
@article{bolder1999yield,
  title={Yield curve modelling at the bank of canada},
  author={Bolder, David Jamieson and Str{\'e}liski, David},
  year={1999},
  publisher={Bank of Canada Technical Report}
}
```

```bibtex
@article{gilli2010calibrating,
  title={Calibrating the nelson-siegel-svensson model},
  author={Gilli, Manfred and Gro{\ss}e, Stefan and Schumann, Enrico},
  journal={Available at SSRN 1676747},
  year={2010}
}
```

