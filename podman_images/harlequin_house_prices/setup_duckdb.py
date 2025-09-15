#!/usr/bin/env python3
"""
Script to create a DuckDB database and load train.csv and test.csv into tables.
"""

import duckdb
import pandas as pd
import os

def create_duckdb_tables():
    """Create DuckDB tables and load CSV data."""

    # Check if CSV files exist
    if not os.path.exists('train.csv'):
        raise FileNotFoundError("train.csv not found in current directory")
    if not os.path.exists('test.csv'):
        raise FileNotFoundError("test.csv not found in current directory")

    # Connect to DuckDB (creates database file if it doesn't exist)
    conn = duckdb.connect('house_prices.db')

    try:
        print("Loading train.csv into DuckDB...")
        # Create train table from CSV
        conn.execute("""
            CREATE OR REPLACE TABLE train AS
            SELECT * FROM read_csv_auto('train.csv')
        """)

        print("Loading test.csv into DuckDB...")
        # Create test table from CSV
        conn.execute("""
            CREATE OR REPLACE TABLE test AS
            SELECT * FROM read_csv_auto('test.csv')
        """)

        # Show table information
        print("\nTrain table info:")
        train_info = conn.execute("DESCRIBE train").fetchall()
        for row in train_info[:5]:  # Show first 5 columns
            print(f"  {row[0]}: {row[1]}")
        if len(train_info) > 5:
            print(f"  ... and {len(train_info) - 5} more columns")

        train_count = conn.execute("SELECT COUNT(*) FROM train").fetchone()[0]
        print(f"  Rows: {train_count}")

        print("\nTest table info:")
        test_info = conn.execute("DESCRIBE test").fetchall()
        for row in test_info[:5]:  # Show first 5 columns
            print(f"  {row[0]}: {row[1]}")
        if len(test_info) > 5:
            print(f"  ... and {len(test_info) - 5} more columns")

        test_count = conn.execute("SELECT COUNT(*) FROM test").fetchone()[0]
        print(f"  Rows: {test_count}")

        print(f"\nDatabase created successfully: house_prices.db")
        print("Tables created: train, test")

    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    create_duckdb_tables()