# FICC

Fixed-Income and Commodities related projects and notebooks 
for learning purposes.

**Notebook list :**
- PV01 and DV01 sensitivity comparison : impact of convexity
- WTI and USDX correlation : empirical confirmation

## PV01 and DV01 for a coupon-paying bond

While : 
$$ 
\text{PV01} = \frac{\partial \text{PV}}{\partial y} \leq 0
$$
values will be considered in absolute term all along.

With a bond paying some coupon on a regular schedule, 
the cash-flows might look like this :

<img src="img/bond_cf.png" width="400">

Different assumptions can be made on the term structure,
either it goes upwards or downwards.

<img src="img/term_structures.png" width="400">

The highest sensitivity, for either of the PV01 or DV01 
goes at the highest maturity :

<img src="img/dv01.png" width="400">

Due to convexity considerations, 
the DV01 will be relatively more important at the tail 
for a downward term structure compared to an upward term structure.

<img src="img/dv01_diff.png" width="400">

(green is positive and red negative in log scale)

Considering the *flat rate* assumption of PV01,
the former plot also involves that if the term structure is **downward sloping**, then (in absolute terms) :

$$
\text{PV01} \leq \text{DV01}
$$

and vice-versa.

## WTI pressured by dollar index

<img src="img/usdx_vs_wti.png" width="400">

The notebook confirms the hypothesis that a stronger US dollar pressures crude oil prices.
Key findings:

- **Negative correlation:** -0.488 (p < 0.01) between WTI spot prices and USDX
- **Explainable variance:** ~33% of WTI variance explained by USDX using VAR model + linear projection
- **Idiosyncratic component:** Remaining 67% represents crude-specific price drivers
- **No lead-lag:** Price adjustments occur intraday (no predictive lag between series)
- **Stationarity:** Log differences are stationary, validating time-series modeling approach

Presence of heavy tail for the USD index :

<img src="img/usdx_wti_qq_plot.png" width="400">

No obvious non-stationarity in the log differenes, then apply stationarity tests to the time series :

<img src="img/usdx_wti_spot_log_diffs.png" width="400">

No obvious autocorrelation to apply (ARMA, VAR to deduce) :

<img src="img/usdx_wti_autocorrelation.png" width="400">

After projecting the USDX on the WTI : 

<img src="img/usdx_wti_linear_projection.png" width="400">

which gives an estimator of the idiosyncratic WTI time-series.

The presence of regime changes can be viewed with a scatter plot with time as a color map :

<img src="img/usdx_wti_market_regime_evolution.png" width="400">

## References :
- Modeles Avances de la Courbe de Taux - Lectures I ... VI, LPSM (M2MO), Zorana Grbac
- https://www.emmi-benchmarks.eu/benchmarks/euribor/rate/
- https://data.ecb.europa.eu/data/data-categories/financial-markets-and-interest-rates/euro-money-market/euro-short-term-rate

