use arrow::record_batch::RecordBatch;
use deltalake::datafusion::prelude::SessionContext;
use deltalake::open_table;
use futures::executor::block_on;
use polars::lazy::dsl::*;
use polars::prelude::*;
use polars::series::ops::NullBehavior;
use std::sync::Arc;
use tokio;

// use polars::lazy::dsl::Expr;// Because polars uses a forked backend of arrow,
// and since type conversion has not been straightforward,
// the arrow to polars_arrow conversion is done by serialisation and deserialisation :
// Arrow -> Parquet -> Polars
// Could also try with IPC support, or conversion to C FFI implementation
fn arrow_to_polars(batch: &RecordBatch) -> Result<DataFrame, Box<dyn std::error::Error>> {
    // Write Arrow batch to parquet file
    let mut buffer = Vec::new();
    {
        use parquet::arrow::ArrowWriter;
        let schema = batch.schema();
        let mut writer = ArrowWriter::try_new(&mut buffer, schema, None)?;
        writer.write(batch)?;
        writer.close()?;
    }

    // Read parquet into Polars
    let cursor = std::io::Cursor::new(buffer);
    let df = polars::prelude::ParquetReader::new(cursor).finish()?;
    Ok(df)
}

// load the given delta table to polars dataframe
fn delta_to_polars(delta_table_path: &str) -> Result<DataFrame, Box<dyn std::error::Error>> {
    // Read and open the delta table
    let table = block_on(async { open_table(delta_table_path).await })?;

    println!("Delta table loaded successfully.");

    // Create a DataFusion context and register the table
    let ctx = SessionContext::new();
    let table_ref = Arc::new(table);

    // Register the Delta table with DataFusion
    block_on(async { ctx.register_table("delta_table", table_ref.clone()) })?;

    let df = block_on(async {
        let query_result = ctx.sql("SELECT * FROM delta_table").await?;
        // Execute the query and collect results
        let batches = query_result.collect().await?;

        if batches.is_empty() {
            return Err("Delta table contains no data".into());
        }
        // Convert record batches to Polars DataFrame
        let mut all_dfs = Vec::new();
        for batch in &batches {
            let df = arrow_to_polars(batch).unwrap();
            all_dfs.push(df);
        }
        print!("{}", all_dfs.get(0).unwrap());

        // Concatenate all dataframes
        if all_dfs.len() == 1 {
            Ok::<DataFrame, Box<dyn std::error::Error>>(all_dfs.remove(0))
        } else {
            let mut result = all_dfs.remove(0);

            // Stack the remaining DataFrames onto the first one
            for df in all_dfs {
                result = result.vstack(&df).expect("Failed to stack DataFrame");
            }
            Ok::<DataFrame, Box<dyn std::error::Error>>(result)
        }
        // Ok(())
    })?;

    Ok(df)
}

fn filter_jumps(df: DataFrame, cutoff: f32) -> Result<DataFrame, PolarsError> {
    // logic to filter jumps
    df.lazy()
        .select([all().exclude(["time"]).backward_fill(None)])
        .with_columns([all()
            .exclude(["time"])
            .diff(1, NullBehavior::Ignore)
            .fill_null(0.)
            .clip((-cutoff).into(), cutoff.into())
            .replace(-cutoff, 0.)
            .replace(cutoff, 0.)
            .cum_sum(false)])
        .collect()
}

// Some dataset for TS prediction
fn multidim_ts_scaled_lookback_flat(
    df: DataFrame,
    target_name: &str,
    scaling_factor: f32,
    lookback: i32,
) -> Result<DataFrame, PolarsError> {
    df.lazy()
        .with_columns([col(target_name).shift(lit(1)).drop_nulls().alias("Y")])
        .with_columns([all()
            .exclude(["time"])
            .diff(1, NullBehavior::Drop)
            .fill_nan(0.)
            .fill_null(0.)
            .drop_nans()
            .drop_nulls()])
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let delta_table_path = "../../raw_data";
    const MAX_DAILY_CHANGE: f32 = 2.0;

    // Read the Delta table and convert to Polars
    let mut polars_df = delta_to_polars(delta_table_path)?;
    polars_df = filter_jumps(polars_df, MAX_DAILY_CHANGE).unwrap();

    println!("Polars DataFrame Head:");
    println!("{}", polars_df.head(Some(5)));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_jumps() {
        let cutoff: f32 = 3.;

        let t = Column::new("time".into(), [1., 2., 3., 4., 5., 6.]);
        let s1e = Column::new("FR01".into(), [0., 1., 2., 3., 3., 5.]);
        let s2e = Column::new("FR02".into(), [0., 0., -2., -2., -1., 1.]);
        let expected_df = DataFrame::new(vec![s1e, s2e]).expect("Could not initialise dataframe");

        let s1 = Column::new("FR01".into(), [1., 2., 3., 4., 0., 2.]);
        let s2 = Column::new("FR02".into(), [0., 5., 3., 0., 1., 3.]);
        let mut df: DataFrame =
            DataFrame::new(vec![t, s1, s2]).expect("Could not initialise dataframe");

        df = filter_jumps(df, cutoff).expect("Filter failed");

        // assert_eq!(df.shape(), expected_df.shape());

        assert!(df.equals(&expected_df));
    }
}
