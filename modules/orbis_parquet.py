
import duckdb
import pandas as pd

def read_parquet(country_iso: str, start:int, end:int) -> pd.DataFrame:
    financials = "/Users/curdinlieberherr/Documents/Schule/HSG/Semester/12.FS26/thesis/industry_global_financials_and_ratios_eur"
    addresses = "/Users/curdinlieberherr/Documents/Schule/HSG/Semester/12.FS26/thesis/all_addresses"
    sectors = "/Users/curdinlieberherr/Documents/Schule/HSG/Semester/12.FS26/thesis/industry_classifications"

    sql = f"""
    WITH addr_filtered AS (
        SELECT DISTINCT bvd_id_number
        FROM read_parquet('{addresses}/*.parquet')
        WHERE country_iso_code = '{country_iso}'
    ),
    sector_filtered AS (
        SELECT DISTINCT bvd_id_number, 
            REPLACE(nace_rev_2_core_code_4_digits_, '.', '') AS nace_code
        FROM read_parquet('{sectors}/*.parquet')
        WHERE TRY_CAST(nace_rev_2_core_code_4_digits_ AS DOUBLE) BETWEEN 10.0 AND 33.0
    ),
    filtered_ids AS (
        SELECT a.bvd_id_number, s.nace_code
        FROM addr_filtered AS a
        JOIN sector_filtered AS s USING (bvd_id_number)
    ),
    fin_filtered AS (
        SELECT fin.*, ids.nace_code,
            ROW_NUMBER() OVER (
                PARTITION BY fin.bvd_id_number, YEAR(closing_date)
                ORDER BY CASE consolidation_code
                    WHEN 'U2' THEN 1
                    WHEN 'U1' THEN 2
                    WHEN 'C2' THEN 3
                    WHEN 'C1' THEN 4
                    WHEN 'LF' THEN 5
                    ELSE 6
                END
            ) AS rn
        FROM read_parquet('{financials}/*.parquet') AS fin
        JOIN filtered_ids ids USING (bvd_id_number)
        WHERE YEAR(closing_date) >= {start} and YEAR(closing_date) <= {end} AND MONTH(closing_date) = 12 AND DAY(closing_date) = 31
    )
    SELECT * EXCLUDE (rn)
    FROM fin_filtered
    WHERE rn = 1
    """

    df = duckdb.query(sql).df()

    return df

def partition_db_to_country_files():
    import duckdb
    import os
    financials = "/Users/curdinlieberherr/Documents/Schule/HSG/Semester/12.FS26/thesis/industry_global_financials_and_ratios_eur"
    addresses = "/Users/curdinlieberherr/Documents/Schule/HSG/Semester/12.FS26/thesis/all_addresses"
    sectors = "/Users/curdinlieberherr/Documents/Schule/HSG/Semester/12.FS26/thesis/industry_classifications"
    output_folder = "/Users/curdinlieberherr/Documents/Schule/HSG/Semester/12.FS26/thesis/financials_per_country/"

    con = duckdb.connect()
    con.execute("PRAGMA max_temp_directory_size='30GiB'")
    countries = ['ES','BE', 'HR', 'BG', 'PL', 'PT', 'DE', 'SE']

    for country_iso in countries:
        path = os.path.join(output_folder, f"{country_iso}.parquet")
        if os.path.exists(path):
            print(f"{country_iso}: bereits vorhanden, überspringe.")
            continue
        sql = f"""
        COPY (
            WITH addr_filtered AS (
                SELECT DISTINCT bvd_id_number
                FROM read_parquet('{addresses}/*.parquet')
                WHERE country_iso_code = '{country_iso}'
            ),
            sector_filtered AS (
                SELECT DISTINCT bvd_id_number, 
                    REPLACE(nace_rev_2_core_code_4_digits_, '.', '') AS nace_code
                FROM read_parquet('{sectors}/*.parquet')
            ),
            filtered_ids AS (
                SELECT a.bvd_id_number, s.nace_code
                FROM addr_filtered AS a
                JOIN sector_filtered AS s USING (bvd_id_number)
            ),
            fin_filtered AS (
                SELECT fin.*, ids.nace_code,
                    ROW_NUMBER() OVER (
                        PARTITION BY fin.bvd_id_number, YEAR(closing_date)
                        ORDER BY CASE consolidation_code
                            WHEN 'U2' THEN 1
                            WHEN 'U1' THEN 2
                            WHEN 'C2' THEN 3
                            WHEN 'C1' THEN 4
                            WHEN 'LF' THEN 5
                            ELSE 6
                        END
                    ) AS rn
                FROM read_parquet('{financials}/*.parquet') AS fin
                JOIN filtered_ids ids USING (bvd_id_number)
                WHERE MONTH(closing_date) = 12 AND DAY(closing_date) = 31
            )
            SELECT * EXCLUDE (rn)
            FROM fin_filtered
            WHERE rn = 1
        )
        TO '{path}'
        (FORMAT PARQUET)
        """
        con.execute(sql)
        print(f"✓ {country_iso} done")

def read_from_partitioned_file(country_iso: str, start:int, end:int, sectormin:int=10, sectormax:int=33) -> pd.DataFrame:
    source_file = f'/Users/curdinlieberherr/Documents/Schule/HSG/Semester/12.FS26/thesis/financials_per_country/{country_iso}.parquet'
    sql = f"""
        SELECT * FROM read_parquet('{source_file}')
        WHERE YEAR(closing_date) >= {start} and YEAR(closing_date) <= {end} AND TRY_CAST(LEFT(nace_code, 2) AS DOUBLE) BETWEEN {sectormin} AND {sectormax}
        """
    df = duckdb.query(sql).df()
    
    return df