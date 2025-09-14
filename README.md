# Fixed Income backend and analytics

With public data related to rates, macro, commodities and crypto, 
the project aims to approach systematic trading practices 
from data ingestion to data analytics.


### Guidelines
- Well-defined services with reproducible environments,
- Accountable and safe data processing from different sources,
- Quantified analytics, model comparisons,

### Techniques
- REST or gRPC API
- Data Lake / Lake House architecture `to be decided`,
- Containerisation with `podman`,
- Managing Python dependencies with `uv`, integrating Rust code,
- `Github workflow` for CI.

### Algos
- Tabular and time-series off-the-shelf ML,
    -  `sklearn`
- Market mode modelisation

### Objective structure

<img src="img/process_flowchart.png" width="600">


## Data
Challenge to unify data from different sources,
with different tick frequencies, noise levels and structure. 

- Ingestion pipeline (ETL)
to gather each endpoint (treated individually). 
    - manage exceptions without failing,
    - run in a separate service.

### Options :
- `pandas` / `polars` dataframe for a few datasources only (short term),
- https://datafusion.apache.org/
- https://iceberg.apache.org/
- https://github.com/delta-io/delta-rs?tab=readme-ov-file
- test `duckdb` and `ducklake` to get started : serverless, little configuration
- *external :* https://www.databricks.com/ 


One secondary objective is to implement a test rust service 
with transforming logic written in `polars`, 
while the querying in the tables can be done with `datafusion`
and placed in a `delta-lake` table.

> [!WARNING]
> Converting from native arrow to polars' version of arrow has been a rough-ride
> due to conversion deprecation
> and dependency conflicts from [here](https://github.com/delta-io/delta-rs/issues/3391).
> Temporary intermediate conversion to parquet.


## Targets

**Covariance matrix estimation**
For clustering (in terms of rates, or in terms of days to define market modes).

**Crypto Predictive Variable**
Is a [Kaggle competition](https://www.kaggle.com/competitions/drw-crypto-market-prediction/data) 
where the objective is Pearson's correlation coefficient.

## Models
- XGBoost Regressor fine-tuning,
- MacMahon, Mel, and Diego Garlaschelli. "Community detection for correlation matrices." arXiv preprint arXiv:1311.1924 (2013).

## Project structure

```bash
├── draft
├── img
├── kaggle                 # kaggle submission notebooks
├── podman_images
├── raw_data
│   ├── commo
│   └── crypto
└── src                     # containing both services and libraries
    ├── app                 # API entrypoint 
    │   └── routers
    ├── ingestion           # ETL service
    │   └── src
    ├── models              # analytics helpers
    └── utils
        ├── dataset
        ├── metrics
        ├── test
        └── transforms

```

**Tools overview :**

| Task | Package |
| --- | --- |
| Data preprocessing | polars |
| Linear algebra | numpy |
| Unit tests | pytest |
| API Routing | uvicorn, fastapi |


## Run API

To start the app, run :
```bash
uvicorn src.app.main:app --reload
```

To build the image, run :
```bash
source run.sh
```

## References :
- to display full logging in the uvicorn app :
    https://gist.github.com/liviaerxin/d320e33cbcddcc5df76dd92948e5be3b
- https://github.com/actions/starter-workflows
- https://github.com/astral-sh/uv-docker-example
- https://docs.github.com/en/actions/writing-workflows/about-workflows
- https://github.com/ProsusAI/finBERT?tab=readme-ov-file
***

Data sources :
- https://www.kaggle.com/datasets/everget/government-bonds/data
- https://api.energy-charts.info/
- https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10/references
- https://www.eia.gov/
- https://www.alphavantage.co/
- https://pydata.github.io/pandas-datareader/index.html
- https://developer.yahoo.com/api/ *decomissioned*

***

Other :

*(Flowchart from : )*

![My Skills](https://go-skill-icons.vercel.app/api/icons?i=mermaid)

