
%load_ext autoreload
%autoreload 2
# sys.path.append("D:/Github/note/module")                        # for windows
sys.path.append("/Users/xinc./Documents/GitHub/note")    # for mac
from module.get_info_JQC import GetInfoJQC
from module.get_info_Postgre import GetPostgreData
from module.plot_func import plot, plot_scatter, plot_df_columns, plot_pdf, plot_dropped_positions, plot_sequence
from module.performance_func import summarize_performance, mean_ttest
from module.tools import filter_near_contract
download
client = GetInfoJQC()
pg_uri = "postgresql+psycopg2://devuser:DevPass123!@localhost:5432/spot"
pg_engine = create_engine(pg_uri)
台積電 spot
client.config.database = "QTSE_2025"
symbol = "2330"
start = "2025-01-01"
end = "2025-11-30"

query = f"""
SELECT
    sid, dd, tt, v, dno, io, m, d, cv,
    bp1, bz1, bp2, bz2, bp3, bz3, bp4, bz4, bp5, bz5,
    sp1, sz1, sp2, sz2, sp3, sz3, sp4, sz4, sp5, sz5
FROM dbo.T06
WHERE sid = '{symbol}'
    AND dd BETWEEN '{start}' AND '{end}'
ORDER BY dd, tt
"""

pg_uri = "postgresql+psycopg2://devuser:DevPass123!@localhost:5432/spot"
table_name = f"t{symbol}"
client.export_to_postgre(table_name = table_name, sql_query = query, postgre_uri = pg_uri, if_exists = "replace")
connecting MSSQL server 192.168.0.180 / database QTSE_2025
export to postgre -> postgresql+psycopg2://devuser:DevPass123!@localhost:5432/t2330
python(53852) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Exporting t2330 (chunks):   0%|          | 0/2520 [00:00<?, ?it/s]
Export completed for t2330
# 改 col 名稱

rename_map = {
    "sid": "stock_id",
    "dd": "trade_date",
    "tt": "transaction_time",
    "v": "volume",
    "dno": "declaration_no",
    "io": "in_out",
    "m": "amount",
    "d": "price",
    "cv": "trade_volume",
    "bp1": "bid_1_price",
    "bz1": "bid_1_volume",
    "bp2": "bid_2_price",
    "bz2": "bid_2_volume",
    "bp3": "bid_3_price",
    "bz3": "bid_3_volume",
    "bp4": "bid_4_price",
    "bz4": "bid_4_volume",
    "bp5": "bid_5_price",
    "bz5": "bid_5_volume",
    "sp1": "ask_1_price",
    "sz1": "ask_1_volume",
    "sp2": "ask_2_price",
    "sz2": "ask_2_volume",
    "sp3": "ask_3_price",
    "sz3": "ask_3_volume",
    "sp4": "ask_4_price",
    "sz4": "ask_4_volume",
    "sp5": "ask_5_price",
    "sz5": "ask_5_volume"
}

# 執行 rename
with pg_engine.begin() as conn:
    for old, new in rename_map.items():
        conn.execute(text(f'ALTER TABLE public.{table_name} RENAME COLUMN "{old}" TO "{new}";'))
# 照時間排序

table_name = "t2330"
sorted_table = f"{table_name}_sorted_tmp"

sort_sql = f"""
DROP TABLE IF EXISTS public.{sorted_table};

CREATE TABLE public.{sorted_table} AS
SELECT *
FROM public.{table_name}
ORDER BY trade_date, transaction_time;

ALTER TABLE public.{table_name} RENAME TO {table_name}_unsorted_backup;
ALTER TABLE public.{sorted_table} RENAME TO {table_name};
"""
with pg_engine.begin() as conn:
    conn.execute(text(sort_sql))
鴻海 spot
client.config.database = "QTSE_2025"
symbol = "2317"
start = "2025-01-01"
end = "2025-11-30"

query = f"""
SELECT
    sid, dd, tt, v, dno, io, m, d, cv,
    bp1, bz1, bp2, bz2, bp3, bz3, bp4, bz4, bp5, bz5,
    sp1, sz1, sp2, sz2, sp3, sz3, sp4, sz4, sp5, sz5
FROM dbo.T06
WHERE sid = '{symbol}'
    AND dd BETWEEN '{start}' AND '{end}'
ORDER BY dd, tt
"""

pg_uri = "postgresql+psycopg2://devuser:DevPass123!@localhost:5432/spot"
table_name = f"t{symbol}"
client.export_to_postgre(table_name = table_name, sql_query = query, postgre_uri = pg_uri, if_exists = "replace")
connecting MSSQL server 192.168.0.180 / database QTSE_2025
export to postgre -> postgresql+psycopg2://devuser:DevPass123!@localhost:5432/spot
Exporting t2317 (chunks):   0%|          | 0/4293 [00:00<?, ?it/s]
Export completed for t2317
# 改 col 名稱

rename_map = {
    "sid": "stock_id",
    "dd": "trade_date",
    "tt": "transaction_time",
    "v": "volume",
    "dno": "declaration_no",
    "io": "in_out",
    "m": "amount",
    "d": "price",
    "cv": "trade_volume",
    "bp1": "bid_1_price",
    "bz1": "bid_1_volume",
    "bp2": "bid_2_price",
    "bz2": "bid_2_volume",
    "bp3": "bid_3_price",
    "bz3": "bid_3_volume",
    "bp4": "bid_4_price",
    "bz4": "bid_4_volume",
    "bp5": "bid_5_price",
    "bz5": "bid_5_volume",
    "sp1": "ask_1_price",
    "sz1": "ask_1_volume",
    "sp2": "ask_2_price",
    "sz2": "ask_2_volume",
    "sp3": "ask_3_price",
    "sz3": "ask_3_volume",
    "sp4": "ask_4_price",
    "sz4": "ask_4_volume",
    "sp5": "ask_5_price",
    "sz5": "ask_5_volume"
}

# 執行 rename
with pg_engine.begin() as conn:
    for old, new in rename_map.items():
        conn.execute(text(f'ALTER TABLE public.{table_name} RENAME COLUMN "{old}" TO "{new}";'))
# 照時間排序

sorted_table = f"{table_name}_sorted_tmp"

sort_sql = f"""
DROP TABLE IF EXISTS public.{sorted_table};

CREATE TABLE public.{sorted_table} AS
SELECT *
FROM public.{table_name}
ORDER BY trade_date, transaction_time;

ALTER TABLE public.{table_name} RENAME TO {table_name}_unsorted_backup;
ALTER TABLE public.{sorted_table} RENAME TO {table_name};
"""
with pg_engine.begin() as conn:
    conn.execute(text(sort_sql))
聯發科 spot
client.config.database = "QTSE_2025"
symbol = "2454"
start = "2025-01-01"
end = "2025-11-30"

query = f"""
SELECT
    sid, dd, tt, v, dno, io, m, d, cv,
    bp1, bz1, bp2, bz2, bp3, bz3, bp4, bz4, bp5, bz5,
    sp1, sz1, sp2, sz2, sp3, sz3, sp4, sz4, sp5, sz5
FROM dbo.T06
WHERE sid = '{symbol}'
    AND dd BETWEEN '{start}' AND '{end}'
ORDER BY dd, tt
"""

pg_uri = "postgresql+psycopg2://devuser:DevPass123!@localhost:5432/spot"
table_name = f"t{symbol}"
client.export_to_postgre(table_name = table_name, sql_query = query, postgre_uri = pg_uri, if_exists = "replace")
# 改 col 名稱

rename_map = {
    "sid": "stock_id",
    "dd": "trade_date",
    "tt": "transaction_time",
    "v": "volume",
    "dno": "declaration_no",
    "io": "in_out",
    "m": "amount",
    "d": "price",
    "cv": "trade_volume",
    "bp1": "bid_1_price",
    "bz1": "bid_1_volume",
    "bp2": "bid_2_price",
    "bz2": "bid_2_volume",
    "bp3": "bid_3_price",
    "bz3": "bid_3_volume",
    "bp4": "bid_4_price",
    "bz4": "bid_4_volume",
    "bp5": "bid_5_price",
    "bz5": "bid_5_volume",
    "sp1": "ask_1_price",
    "sz1": "ask_1_volume",
    "sp2": "ask_2_price",
    "sz2": "ask_2_volume",
    "sp3": "ask_3_price",
    "sz3": "ask_3_volume",
    "sp4": "ask_4_price",
    "sz4": "ask_4_volume",
    "sp5": "ask_5_price",
    "sz5": "ask_5_volume"
}

# 執行 rename
with pg_engine.begin() as conn:
    for old, new in rename_map.items():
        conn.execute(text(f'ALTER TABLE public.{table_name} RENAME COLUMN "{old}" TO "{new}";'))
# 照時間排序

sorted_table = f"{table_name}_sorted_tmp"

sort_sql = f"""
DROP TABLE IF EXISTS public.{sorted_table};

CREATE TABLE public.{sorted_table} AS
SELECT *
FROM public.{table_name}
ORDER BY trade_date, transaction_time;

ALTER TABLE public.{table_name} RENAME TO {table_name}_unsorted_backup;
ALTER TABLE public.{sorted_table} RENAME TO {table_name};
"""
with pg_engine.begin() as conn:
    conn.execute(text(sort_sql))
台指期
client.config.database = "QFUT_2025"
months = list("ABCDEFGHIJKL")
f01s = [f"TXF{m}5" for m in months]

start = "2025-01-01"
end   = "2025-11-30"

f01_list_sql = ",".join([f"'{s}'" for s in f01s])
query = f"""
SELECT *
FROM dbo.I020
WHERE f01 IN ({f01_list_sql})
  AND dd BETWEEN '{start}' AND '{end}'
"""

client.export_to_postgre(
    table_name  = "tTXF",
    sql_query   = query,
    postgre_uri = pg_uri,
    if_exists   = "replace",
)

# query = f"""
# SELECT
#     dd  AS trade_date,
#     tt  AS transaction_time,
#     f01 AS sid,
#     udp,
#     f02 AS bid_1_price, f03 AS bid_1_volume,
#     f04 AS bid_2_price, f05 AS bid_2_volume,
#     f06 AS bid_3_price, f07 AS bid_3_volume,
#     f08 AS bid_4_price, f09 AS bid_4_volume,
#     f10 AS bid_5_price, f11 AS bid_5_volume,
#     f12 AS ask_1_price, f13 AS ask_1_volume,
#     f14 AS ask_2_price, f15 AS ask_2_volume,
#     f16 AS ask_3_price, f17 AS ask_3_volume,
#     f18 AS ask_4_price, f19 AS ask_4_volume,
#     f20 AS ask_5_price, f21 AS ask_5_volume,
#     f22 AS has_iceberg,
#     f23 AS iceberg_bid1_price, f24 AS iceberg_bid1_volume,
#     f25 AS iceberg_ask1_price, f26 AS iceberg_ask1_volume,
#     seq
# FROM dbo.I080
# WHERE f01 IN ({f01_list_sql})
#     AND dd BETWEEN '{start}' AND '{end}'
# """
client.export_to_postgre(
    table_name = "i080",
    sql_query = query,
    postgre_uri = pg_uri,
    if_exists = "replace"
)
connecting MSSQL server 192.168.0.180 / database QFUT_2025
export to postgre -> postgresql+psycopg2://devuser:DevPass123!@localhost:5432/spot
target table -> i080
Exporting i080 (chunks):   0%|          | 0/2778 [00:00<?, ?it/s]
---------------------------------------------------------------------------

台積電 future
client.config.database = "QFUT_2025"
months = list("ABCDEFGHIJKL")
cdf_f01s = [f"CDF{m}5" for m in months]

start = "2025-01-01"
end   = "2025-11-30"

f01_list_sql = ",".join([f"'{s}'" for s in cdf_f01s])
query = f"""
SELECT *
FROM dbo.I020
WHERE f01 IN ({f01_list_sql})
  AND dd BETWEEN '{start}' AND '{end}'
"""

client.export_to_postgre(
    table_name  = "CDF",
    sql_query   = query,
    postgre_uri = pg_uri,
    if_exists   = "replace",
)
connecting MSSQL server 192.168.0.180 / database QFUT_2025
export to postgre -> postgresql+psycopg2://devuser:DevPass123!@localhost:5432/t2330
Exporting CDF (chunks):   0%|          | 0/140 [00:00<?, ?it/s]
Export completed for CDF
others
dispose
client.dispose()
pg_engine.dispose()
featurize
t2330
resample & featurize to ms
pg_engine = create_engine("postgresql+psycopg2://devuser:DevPass123!@localhost:5432/spot")

# 可自訂輸出表名
target_schema = "public"
target_table  = "ms50_2330"
target_qualified = f"{target_schema}.{target_table}"

start_date  = "2025-01-01"
end_date    = "2025-11-30"
start_time  = "13:00:00"
end_time    = "13:30:00"
cutoff_time = "13:24:00"
bucket_ms   = 50
chunk_days  = 3

create_sql = f"""
CREATE TABLE {target_qualified} AS
WITH date_range AS (
    SELECT DISTINCT trade_date
    FROM public.t2330
    WHERE trade_date BETWEEN :chunk_start AND :chunk_end
),
source_data AS (
    SELECT trade_date, transaction_time, trade_volume, in_out,
           bid_1_price, bid_1_volume, bid_2_price, bid_2_volume,
           ask_1_price, ask_1_volume, ask_2_price, ask_2_volume,
           NULLIF(price,0) AS price,
           (trade_date::timestamp
             + (floor(EXTRACT(EPOCH FROM transaction_time::time)*1000/:bucket_ms)::bigint
                * :bucket_ms * INTERVAL '1 millisecond'))::timestamp AS bucket_start
    FROM public.t2330
    WHERE trade_date BETWEEN :chunk_start AND :chunk_end
      AND transaction_time::time >= CAST(:start_time AS time)
      AND (:end_time IS NULL OR transaction_time::time <= CAST(:end_time AS time))
),
bucket_series AS (
    SELECT d.trade_date, gs.bucket_start::timestamp AS bucket_start
    FROM date_range d
    CROSS JOIN LATERAL generate_series(
        d.trade_date::timestamp + CAST(:start_time AS time),
        d.trade_date::timestamp + CAST(:end_time   AS time),
        (:bucket_ms)::int * INTERVAL '1 millisecond'
    ) AS gs(bucket_start)
),
per_bucket AS (
    SELECT trade_date, bucket_start,
           SUM(trade_volume) AS volume,
           SUM(in_out)       AS sum_in_out
    FROM source_data
    GROUP BY trade_date, bucket_start
),
latest_tick AS (
    SELECT trade_date, bucket_start, price,
           bid_1_price, bid_1_volume, bid_2_price, bid_2_volume,
           ask_1_price, ask_1_volume, ask_2_price, ask_2_volume
    FROM (
        SELECT sd.*,
               ROW_NUMBER() OVER (PARTITION BY sd.trade_date, sd.bucket_start
                                  ORDER BY sd.transaction_time DESC) AS rn
        FROM source_data sd
    ) t WHERE rn = 1
),
locf_prep AS (
    SELECT
        bs.trade_date, bs.bucket_start,
        lt.price, lt.bid_1_price, lt.bid_1_volume, lt.bid_2_price, lt.bid_2_volume,
        lt.ask_1_price, lt.ask_1_volume, lt.ask_2_price, lt.ask_2_volume,
        pb.volume, pb.sum_in_out,
        ROW_NUMBER() OVER w AS seq,
        SUM(CASE WHEN lt.price        IS NOT NULL THEN 1 ELSE 0 END) OVER w AS grp_p,
        SUM(CASE WHEN lt.bid_1_price  IS NOT NULL THEN 1 ELSE 0 END) OVER w AS grp_b1,
        SUM(CASE WHEN lt.bid_1_volume IS NOT NULL THEN 1 ELSE 0 END) OVER w AS grp_b1v,
        SUM(CASE WHEN lt.bid_2_price  IS NOT NULL THEN 1 ELSE 0 END) OVER w AS grp_b2,
        SUM(CASE WHEN lt.bid_2_volume IS NOT NULL THEN 1 ELSE 0 END) OVER w AS grp_b2v,
        SUM(CASE WHEN lt.ask_1_price  IS NOT NULL THEN 1 ELSE 0 END) OVER w AS grp_a1,
        SUM(CASE WHEN lt.ask_1_volume IS NOT NULL THEN 1 ELSE 0 END) OVER w AS grp_a1v,
        SUM(CASE WHEN lt.ask_2_price  IS NOT NULL THEN 1 ELSE 0 END) OVER w AS grp_a2,
        SUM(CASE WHEN lt.ask_2_volume IS NOT NULL THEN 1 ELSE 0 END) OVER w AS grp_a2v
    FROM bucket_series bs
    LEFT JOIN latest_tick lt
      ON lt.trade_date = bs.trade_date AND lt.bucket_start = bs.bucket_start
    LEFT JOIN per_bucket pb
      ON pb.trade_date = bs.trade_date AND pb.bucket_start = bs.bucket_start
    WINDOW w AS (PARTITION BY bs.trade_date ORDER BY bs.bucket_start)
),
locf AS (
    SELECT
        trade_date, bucket_start, volume, sum_in_out,
        FIRST_VALUE(price)        OVER (PARTITION BY trade_date, grp_p  ORDER BY seq) AS last_price,
        FIRST_VALUE(bid_1_price)  OVER (PARTITION BY trade_date, grp_b1 ORDER BY seq) AS bid_1_price,
        FIRST_VALUE(bid_1_volume) OVER (PARTITION BY trade_date, grp_b1v ORDER BY seq) AS bid_1_volume,
        FIRST_VALUE(bid_2_price)  OVER (PARTITION BY trade_date, grp_b2 ORDER BY seq) AS bid_2_price,
        FIRST_VALUE(bid_2_volume) OVER (PARTITION BY trade_date, grp_b2v ORDER BY seq) AS bid_2_volume,
        FIRST_VALUE(ask_1_price)  OVER (PARTITION BY trade_date, grp_a1 ORDER BY seq) AS ask_1_price,
        FIRST_VALUE(ask_1_volume) OVER (PARTITION BY trade_date, grp_a1v ORDER BY seq) AS ask_1_volume,
        FIRST_VALUE(ask_2_price)  OVER (PARTITION BY trade_date, grp_a2 ORDER BY seq) AS ask_2_price,
        FIRST_VALUE(ask_2_volume) OVER (PARTITION BY trade_date, grp_a2v ORDER BY seq) AS ask_2_volume
    FROM locf_prep
),
cutoff_price AS (
    SELECT trade_date, last_price AS cutoff_price
    FROM (
        SELECT trade_date, bucket_start, last_price,
               ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY bucket_start DESC) AS rn
        FROM locf
        WHERE bucket_start::time <= CAST(:cutoff_time AS time)
    ) t WHERE rn = 1
),
vwap_20_24 AS (
    SELECT trade_date,
           SUM(COALESCE(last_price,0) * COALESCE(volume,0)) / NULLIF(SUM(COALESCE(volume,0)),0) AS vwap_20_24
    FROM locf
    WHERE bucket_start::time >= CAST('13:20:00' AS time)
      AND bucket_start::time <  CAST('13:24:00' AS time)
    GROUP BY trade_date
),
close_price AS (
    SELECT trade_date, last_price AS close_price
    FROM (
        SELECT trade_date, bucket_start, last_price,
               ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY bucket_start DESC) AS rn
        FROM locf
    ) t
    WHERE rn = 1
),
label_vwap AS (
    SELECT c.trade_date,
           CASE
               WHEN v.vwap_20_24 IS NULL OR v.vwap_20_24 = 0 THEN 0
               WHEN c.close_price > v.vwap_20_24 THEN 1
               WHEN c.close_price < v.vwap_20_24 THEN -1
               ELSE 0
           END AS vwap_label
    FROM close_price c
    LEFT JOIN vwap_20_24 v ON v.trade_date = c.trade_date
),
final AS (
    SELECT
        locf.trade_date, locf.bucket_start,
        COALESCE(locf.volume, 0)::bigint     AS volume,
        COALESCE(locf.sum_in_out, 0)::bigint AS sum_in_out,
        COALESCE(locf.last_price, 0)         AS last_price,
        CASE WHEN cp.cutoff_price IS NULL THEN 0
             WHEN locf.last_price > cp.cutoff_price THEN 1
             WHEN locf.last_price < cp.cutoff_price THEN -1
             ELSE 0 END AS relative_rise_fall,
        COALESCE(lv.vwap_label,0) AS relative_vwap_rise_fall,
        GREATEST(0, EXTRACT(EPOCH FROM (locf.trade_date::timestamp + CAST(:end_time AS time) - locf.bucket_start))) AS time_to_close_s,
        COALESCE(locf.last_price - LAG(locf.last_price) OVER ret_w, 0) AS price_delta,
        CASE WHEN LAG(last_price,20) OVER ret_w IS NULL OR LAG(last_price,20) OVER ret_w=0 THEN NULL
             ELSE last_price / LAG(last_price,20) OVER ret_w - 1 END AS return_1s,
        CASE WHEN LAG(last_price,100) OVER ret_w IS NULL OR LAG(last_price,100) OVER ret_w=0 THEN NULL
             ELSE last_price / LAG(last_price,100) OVER ret_w - 1 END AS return_5s,
        STDDEV_SAMP(last_price) OVER (ret_w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS vol_1s,
        STDDEV_SAMP(last_price) OVER (ret_w ROWS BETWEEN 99 PRECEDING AND CURRENT ROW) AS vol_5s,
        SUM(COALESCE(volume,0)) OVER (ret_w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS volume_1s,
        SUM(COALESCE(volume,0)) OVER (ret_w ROWS BETWEEN 99 PRECEDING AND CURRENT ROW) AS volume_5s,
        CASE WHEN SUM(COALESCE(volume,0)) OVER (ret_w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)=0 THEN 0
             ELSE SUM(COALESCE(sum_in_out,0)) OVER (ret_w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
                  / SUM(COALESCE(volume,0)) OVER (ret_w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) END AS io_imb_1s,
        CASE WHEN SUM(COALESCE(volume,0)) OVER (ret_w ROWS BETWEEN 99 PRECEDING AND CURRENT ROW)=0 THEN 0
             ELSE SUM(COALESCE(sum_in_out,0)) OVER (ret_w ROWS BETWEEN 99 PRECEDING AND CURRENT ROW)
                  / SUM(COALESCE(volume,0)) OVER (ret_w ROWS BETWEEN 99 PRECEDING AND CURRENT ROW) END AS io_imb_5s,
        COALESCE(bid_1_price,0)          AS bid_1_price,
        COALESCE(bid_1_volume,0)::bigint AS bid_1_volume,
        COALESCE(bid_1_volume - LAG(bid_1_volume) OVER ret_w,0)::bigint AS bid_1_volume_delta,
        COALESCE(bid_2_price,0)          AS bid_2_price,
        COALESCE(bid_2_volume,0)::bigint AS bid_2_volume,
        COALESCE(ask_1_price,0)          AS ask_1_price,
        COALESCE(ask_1_volume,0)::bigint AS ask_1_volume,
        COALESCE(ask_1_volume - LAG(ask_1_volume) OVER ret_w,0)::bigint AS ask_1_volume_delta,
        COALESCE(ask_2_price,0)          AS ask_2_price,
        COALESCE(ask_2_volume,0)::bigint AS ask_2_volume,
        CASE WHEN (COALESCE(bid_1_volume,0)+COALESCE(ask_1_volume,0))=0 THEN 0
             ELSE (COALESCE(bid_1_volume,0)-COALESCE(ask_1_volume,0))::numeric
                  / NULLIF((COALESCE(bid_1_volume,0)+COALESCE(ask_1_volume,0))::numeric,0) END AS depth_imb
    FROM locf
    LEFT JOIN cutoff_price cp ON cp.trade_date = locf.trade_date
    LEFT JOIN label_vwap lv ON lv.trade_date = locf.trade_date
    WINDOW ret_w AS (PARTITION BY locf.trade_date ORDER BY locf.bucket_start)
)
SELECT * FROM final
ORDER BY trade_date, bucket_start;
"""

# 後續 chunk 插入（同樣 SQL，只是 INSERT INTO）
insert_sql = create_sql.replace(f"CREATE TABLE {target_qualified} AS", f"INSERT INTO {target_qualified}")

params_common = {
    "start_time": start_time,
    "end_time": end_time,
    "cutoff_time": cutoff_time,
    "bucket_ms": bucket_ms,
}
with pg_engine.begin() as conn:
    conn.exec_driver_sql(f"DROP TABLE IF EXISTS {target_qualified}")


first_chunk = True
for chunk_start in tqdm(pd.date_range(start_date, end_date, freq=f"{chunk_days}D")):
    cs = chunk_start.date()
    ce = min(pd.to_datetime(cs) + pd.Timedelta(days=chunk_days-1), pd.to_datetime(end_date)).date()
    sql = create_sql if first_chunk else insert_sql
    with pg_engine.begin() as conn:
        conn.execute(text(sql), {**params_common, "chunk_start": cs, "chunk_end": ce})
    first_chunk = False

print("Done: rebuilt public.ms50_2330。")
python(14316) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
  0%|          | 0/112 [00:00<?, ?it/s]

featurize
# Tick-level rebuild: t2330 -> t2330_features (no resample) with quote/depth/order-flow labels
# Optimized with Indexing and Parallel Processing
# (Clean Event-Based Logic, No Extra Features from Cell 34)
from sqlalchemy import create_engine, text
from tqdm.auto import tqdm
from datetime import datetime, timedelta
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing

src_qualified = "public.t2330"
tgt_qualified = "public.t2330_features"

# Database connection string
db_uri = "postgresql+psycopg2://devuser:DevPass123!@localhost:5432/spot"
pg_engine = create_engine(db_uri, future=True)

start_date  = "2025-01-01"
end_date    = "2025-11-30"
start_time  = "13:00:00"
end_time    = "13:30:00"  # upper bound; set None to ignore
cutoff_time = "13:24:00"
chunk_days  = 5  # Small chunk size for better load balancing in parallel

# -------------------------------
# 1. Create Index for Performance
# -------------------------------
print("Ensuring index exists on source table...")
with pg_engine.begin() as conn:
    conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_t2330_date_time ON {src_qualified} (trade_date, transaction_time)"))

# -------------------------------
# SQL: tick-level features + labels (no resample)
# -------------------------------
sql_base = f"""
WITH src AS (
    SELECT *
    FROM {src_qualified}
    WHERE trade_date BETWEEN :s AND :e
      AND transaction_time::time >= CAST(:start_time AS time)
      AND (:end_time IS NULL OR transaction_time::time <= CAST(:end_time AS time))
),
mid_spread AS (
    SELECT
        src.*,
        (bid_1_price + ask_1_price)/2.0 AS mid_price,
        (ask_1_price - bid_1_price)     AS spread,
        LAG((bid_1_price + ask_1_price)/2.0)
            OVER (PARTITION BY trade_date ORDER BY transaction_time) AS lag_mid_price,
        LAG(ask_1_price - bid_1_price)
            OVER (PARTITION BY trade_date ORDER BY transaction_time) AS lag_spread,
        -- Identify trade events for LOCF
        CASE WHEN trade_volume > 0 THEN price ELSE NULL END AS raw_trade_price,
        SUM(CASE WHEN trade_volume > 0 THEN 1 ELSE 0 END)
            OVER (PARTITION BY trade_date ORDER BY transaction_time) AS trade_grp
    FROM src
),
with_last_price AS (
    SELECT
        ms.*,
        -- LOCF: fill forward the last valid trade price
        FIRST_VALUE(raw_trade_price) OVER (PARTITION BY trade_date, trade_grp ORDER BY transaction_time) AS last_trade_price
    FROM mid_spread ms
),
computed AS (
    SELECT
        wp.*,
        -- last change timestamps (no nested windows)
        MAX(CASE WHEN mid_price IS DISTINCT FROM lag_mid_price THEN transaction_time END)
            OVER (PARTITION BY trade_date ORDER BY transaction_time
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS last_mid_change_ts,
        MAX(CASE WHEN spread   IS DISTINCT FROM lag_spread   THEN transaction_time END)
            OVER (PARTITION BY trade_date ORDER BY transaction_time
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS last_spread_change_ts,
        (bid_1_volume - LAG(bid_1_volume) OVER (PARTITION BY trade_date ORDER BY transaction_time))::double precision AS delta_bid_1_volume,
        (ask_1_volume - LAG(ask_1_volume) OVER (PARTITION BY trade_date ORDER BY transaction_time))::double precision AS delta_ask_1_volume,
        (bid_2_volume - LAG(bid_2_volume) OVER (PARTITION BY trade_date ORDER BY transaction_time))::double precision AS delta_bid_2_volume,
        (ask_2_volume - LAG(ask_2_volume) OVER (PARTITION BY trade_date ORDER BY transaction_time))::double precision AS delta_ask_2_volume,
        CASE WHEN (bid_1_volume + ask_1_volume) = 0 THEN 0
             ELSE (bid_1_volume - ask_1_volume)::double precision / NULLIF((bid_1_volume + ask_1_volume)::double precision, 0)
        END AS depth_imb,
        CASE WHEN (bid_1_volume + bid_2_volume + ask_1_volume + ask_2_volume) = 0 THEN 0
             ELSE ((bid_1_volume + bid_2_volume)-(ask_1_volume + ask_2_volume))::double precision
                  / NULLIF((bid_1_volume + bid_2_volume + ask_1_volume + ask_2_volume)::double precision, 0)
        END AS depth_imb_5,
        (SUM(in_out) OVER (
            PARTITION BY trade_date ORDER BY transaction_time
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ))::double precision AS in_out_rolling_5,
        (SUM(in_out) OVER (
            PARTITION BY trade_date ORDER BY transaction_time
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ))::double precision AS in_out_rolling_20
    FROM with_last_price wp
),
cutoff_price AS (
    SELECT trade_date, last_trade_price AS cutoff_price
    FROM (
        SELECT trade_date, last_trade_price, transaction_time,
               ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY transaction_time DESC) AS rn
        FROM computed
        WHERE transaction_time::time <= CAST(:cutoff_time AS time)
              AND last_trade_price IS NOT NULL AND last_trade_price <> 0
    ) t WHERE rn = 1
),
vwap_20_24 AS (
    SELECT trade_date,
           SUM(COALESCE(last_trade_price,0) * COALESCE(trade_volume,0)) / NULLIF(SUM(COALESCE(trade_volume,0)),0) AS vwap_20_24
    FROM computed
    WHERE transaction_time::time >= CAST('13:20:00' AS time)
      AND transaction_time::time <  CAST('13:24:00' AS time)
    GROUP BY trade_date
),
close_price AS (
    SELECT trade_date, last_trade_price AS close_price
    FROM (
        SELECT trade_date, last_trade_price, transaction_time,
               ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY transaction_time DESC) AS rn
        FROM computed
        WHERE last_trade_price IS NOT NULL AND last_trade_price <> 0
    ) t WHERE rn = 1
),
label_vwap AS (
    SELECT c.trade_date,
           CASE
               WHEN v.vwap_20_24 IS NULL OR v.vwap_20_24 = 0 THEN 0
               WHEN c.close_price > v.vwap_20_24 THEN 1
               WHEN c.close_price < v.vwap_20_24 THEN -1
               ELSE 0
           END AS relative_vwap_rise_fall
    FROM close_price c
    LEFT JOIN vwap_20_24 v ON v.trade_date = c.trade_date
)
SELECT
    c.*,  -- all original + mid/spread + deltas/rolling
    EXTRACT(EPOCH FROM (transaction_time - COALESCE(last_mid_change_ts, transaction_time))) AS mid_price_time_since_change,
    EXTRACT(EPOCH FROM (transaction_time - COALESCE(last_spread_change_ts, transaction_time))) AS spread_time_since_change,
    -- labels & time-based features
    CASE WHEN cp.cutoff_price IS NULL THEN 0
         WHEN c.last_trade_price > cp.cutoff_price THEN 1
         WHEN c.last_trade_price < cp.cutoff_price THEN -1
         ELSE 0 END AS relative_rise_fall,
    COALESCE(lv.relative_vwap_rise_fall, 0) AS relative_vwap_rise_fall,
    GREATEST(0, EXTRACT(EPOCH FROM ((c.trade_date::timestamp + CAST(:end_time AS time)) - c.transaction_time))) AS time_to_close_s,
    
    -- Event-based features (using last_trade_price and trade_volume)
    COALESCE(c.last_trade_price - LAG(c.last_trade_price) OVER ret_w, 0) AS price_delta,
    
    -- Return 20 events
    CASE WHEN LAG(c.last_trade_price, 20) OVER ret_w IS NULL OR LAG(c.last_trade_price, 20) OVER ret_w = 0 THEN NULL
         ELSE c.last_trade_price / LAG(c.last_trade_price, 20) OVER ret_w - 1 END AS return_20ev,
         
    -- Return 100 events
    CASE WHEN LAG(c.last_trade_price, 100) OVER ret_w IS NULL OR LAG(c.last_trade_price, 100) OVER ret_w = 0 THEN NULL
         ELSE c.last_trade_price / LAG(c.last_trade_price, 100) OVER ret_w - 1 END AS return_100ev,
         
    -- Volatility (stddev of last_trade_price over last 20/100 events)
    STDDEV_SAMP(c.last_trade_price) OVER (ret_w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS vol_20ev,
    STDDEV_SAMP(c.last_trade_price) OVER (ret_w ROWS BETWEEN 99 PRECEDING AND CURRENT ROW) AS vol_100ev,
    
    -- Rolling Trade Volume (sum of trade_volume over last 20/100 events)
    SUM(COALESCE(c.trade_volume,0)) OVER (ret_w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS trade_vol_20ev,
    SUM(COALESCE(c.trade_volume,0)) OVER (ret_w ROWS BETWEEN 99 PRECEDING AND CURRENT ROW) AS trade_vol_100ev,
    
    -- IO Imbalance (using trade_volume instead of volume)
    CASE WHEN SUM(COALESCE(c.trade_volume,0)) OVER (ret_w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) = 0 THEN 0
         ELSE SUM(COALESCE(c.in_out,0)) OVER (ret_w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
              / SUM(COALESCE(c.trade_volume,0)) OVER (ret_w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
    END AS io_imb_20ev,
    
    CASE WHEN SUM(COALESCE(c.trade_volume,0)) OVER (ret_w ROWS BETWEEN 99 PRECEDING AND CURRENT ROW) = 0 THEN 0
         ELSE SUM(COALESCE(c.in_out,0)) OVER (ret_w ROWS BETWEEN 99 PRECEDING AND CURRENT ROW)
              / SUM(COALESCE(c.trade_volume,0)) OVER (ret_w ROWS BETWEEN 99 PRECEDING AND CURRENT ROW)
    END AS io_imb_100ev
    
FROM computed c
LEFT JOIN cutoff_price cp ON cp.trade_date = c.trade_date
LEFT JOIN label_vwap lv   ON lv.trade_date = c.trade_date
WINDOW ret_w AS (PARTITION BY c.trade_date ORDER BY c.transaction_time)
ORDER BY c.trade_date, c.transaction_time
"""

create_stmt = text(f"CREATE TABLE {tgt_qualified} AS " + sql_base)
insert_stmt = text(f"INSERT INTO {tgt_qualified} " + sql_base)

# -------------------------------
# chunk helper
# -------------------------------
def chunks(s, e, days):
    cur = datetime.fromisoformat(s).date()
    end = datetime.fromisoformat(e).date()
    while cur <= end:
        nxt = min(cur + timedelta(days=days-1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)

all_chunks = list(chunks(start_date, end_date, chunk_days))
first_chunk = all_chunks[0]
remaining_chunks = all_chunks[1:]

# -------------------------------
# 2. Process first chunk (Create Table)
# -------------------------------
print(f"Processing first chunk {first_chunk[0]} to {first_chunk[1]} (Creating Table)...")
with pg_engine.begin() as conn:
    conn.exec_driver_sql(f"DROP TABLE IF EXISTS {tgt_qualified}")
    conn.execute(create_stmt, {"s": first_chunk[0], "e": first_chunk[1], "start_time": start_time, "end_time": end_time, "cutoff_time": cutoff_time})

# -------------------------------
# 3. Process remaining chunks (Parallel Insert)
# -------------------------------
def process_chunk(chunk):
    s, e = chunk
    # Create a new engine/connection per process to be safe
    eng = create_engine(db_uri, future=True)
    with eng.begin() as conn:
        conn.execute(insert_stmt, {"s": s, "e": e, "start_time": start_time, "end_time": end_time, "cutoff_time": cutoff_time})
    eng.dispose()

if remaining_chunks:
    print(f"Processing {len(remaining_chunks)} chunks in parallel...")
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    Parallel(n_jobs=n_jobs)(delayed(process_chunk)(chunk) for chunk in tqdm(remaining_chunks))

with pg_engine.connect() as conn:
    total_rows = conn.execute(text(f"SELECT COUNT(*) FROM {tgt_qualified}")).scalar()
print(f"✔ Done: {tgt_qualified} built. Total rows: {total_rows}")
Ensuring index exists on source table...
Processing first chunk 2025-01-01 to 2025-01-05 (Creating Table)...
Processing 66 chunks in parallel...
python(35282) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
  0%|          | 0/66 [00:00<?, ?it/s]

✔ Done: public.t2330_features built. Total rows: 1507327
# -----------------------------------------------------------------------------
# Merge tTXF (Futures) into t2330_features (Interleave & Forward Fill)
# -----------------------------------------------------------------------------
from sqlalchemy import create_engine, text
import pandas as pd
from tqdm.auto import tqdm

# --- Configuration ---
stock_table  = "public.t2330_features"       # Source stock table (in spot DB)
future_table = 'public."tTXF"'               # Source future table (in future DB)
target_table = "public.t2330_features_final" # Target table (in spot DB)

# Define TWO database connections
db_uri_spot   = "postgresql+psycopg2://devuser:DevPass123!@localhost:5432/spot"
db_uri_future = "postgresql+psycopg2://devuser:DevPass123!@localhost:5432/future"

engine_spot   = create_engine(db_uri_spot, future=True)
engine_future = create_engine(db_uri_future, future=True)

# 1. Get distinct dates to process (from Stock DB)
print("Fetching trade dates from Spot DB...")
with engine_spot.connect() as conn:
    dates_df = pd.read_sql(f"SELECT DISTINCT trade_date FROM {stock_table} ORDER BY trade_date", conn)
    dates = dates_df['trade_date'].astype(str).tolist()

# Verify Future DB connection
print("Verifying Future DB connection...")
try:
    with engine_future.connect() as conn:
        conn.execute(text(f"SELECT 1 FROM {future_table} LIMIT 1"))
        print(f"✔ Found {future_table} in Future DB.")
except Exception as e:
    print(f"❌ Error connecting to Future DB or finding table: {e}")
    raise e

print(f"Processing {len(dates)} days...")

# 2. Process in chunks
chunk_size = 5
for i in tqdm(range(0, len(dates), chunk_size), desc="Interleaving Data"):
    chunk_dates = dates[i : i + chunk_size]
    if not chunk_dates:
        continue
        
    s, e = chunk_dates[0], chunk_dates[-1]
    
    try:
        # A. Read Stock Features (from Spot DB)
        q_stock = f"SELECT * FROM {stock_table} WHERE trade_date BETWEEN '{s}' AND '{e}' ORDER BY trade_date, transaction_time"
        df_stock = pd.read_sql(q_stock, engine_spot)
        
        # Create full datetime column for sorting
        df_stock['ts'] = pd.to_datetime(
            df_stock['trade_date'].astype(str) + ' ' + df_stock['transaction_time'].astype(str),
            format='mixed'
        )
        df_stock['data_source'] = 'stock'
        
        # B. Read Futures Data (from Future DB)
        q_future = f"""
        SELECT 
            dd as trade_date, 
            tt as transaction_time, 
            f04 as txf_price, 
            f05 as txf_volume,
            f04 - LAG(f04, 20) OVER (PARTITION BY dd ORDER BY tt) as txf_price_delta_20ev,
            f04 - LAG(f04, 100) OVER (PARTITION BY dd ORDER BY tt) as txf_price_delta_100ev,
            f05 - LAG(f05, 20) OVER (PARTITION BY dd ORDER BY tt) as txf_volume_delta_20ev,
            f05 - LAG(f05, 100) OVER (PARTITION BY dd ORDER BY tt) as txf_volume_delta_100ev
        FROM {future_table} 
        WHERE dd BETWEEN '{s}' AND '{e}' 
        ORDER BY dd, tt
        """
        df_future = pd.read_sql(q_future, engine_future)
        
        # Create full datetime column for sorting
        df_future['ts'] = pd.to_datetime(
            df_future['trade_date'].astype(str) + ' ' + df_future['transaction_time'].astype(str),
            format='mixed'
        )
        df_future['data_source'] = 'future'
        
        # C. Interleave (Concat + Sort)
        # Combine both datasets
        df_combined = pd.concat([df_stock, df_future], ignore_index=True)
        
        # Sort by timestamp
        df_combined = df_combined.sort_values('ts')
        
        # D. Forward Fill
        # Propagate last valid observation forward to next valid
        df_combined = df_combined.ffill()
        
        # Drop the temporary 'ts' column
        df_combined = df_combined.drop(columns=['ts'])
        
        # E. Write to DB (Spot DB)
        if_exists = 'replace' if i == 0 else 'append'
        df_combined.to_sql(
            target_table.split('.')[1], 
            engine_spot, 
            schema='public', 
            if_exists=if_exists, 
            index=False, 
            method='multi', 
            chunksize=2000
        )
        
    except Exception as exc:
        print(f"❌ Error processing chunk {s} to {e}: {exc}")

print(f"✔ Done. Final table: {target_table}")
Fetching trade dates from Spot DB...
Verifying Future DB connection...
✔ Found public."tTXF" in Future DB.
Processing 201 days...
Interleaving Data:   0%|          | 0/41 [00:00<?, ?it/s]
鴻海
pg_uri = "postgresql+psycopg2://devuser:DevPass123!@localhost:5432/spot"
pg_engine = create_engine(pg_uri, future=True)

start_date = "2025-01-01"
end_date   = "2025-11-30"
start_time = "13:00:00"
end_time: str | None = None  # 若需上界可設定，如 "13:30:00"

trade_dates_query = text("""
SELECT DISTINCT trade_date
FROM public.t2317
WHERE trade_date BETWEEN :start_date AND :end_date
ORDER BY trade_date
""")

time_upper_clause = "AND transaction_time::time <= :end_time" if end_time else ""
day_query = text(f"""
SELECT *
FROM public.t2317
WHERE trade_date = :trade_date
  AND transaction_time::time >= :start_time
  {time_upper_clause}
ORDER BY transaction_time
""")

volume_cols = [
    "bid_1_volume", "ask_1_volume",
    "bid_2_volume", "ask_2_volume",
    "bid_3_volume", "ask_3_volume",
    "bid_4_volume", "ask_4_volume",
    "bid_5_volume", "ask_5_volume",
    "volume",
]

with pg_engine.connect() as conn:
    trade_dates = pd.read_sql_query(
        trade_dates_query,
        conn,
        params = {"start_date": start_date, "end_date": end_date},
    )["trade_date"].tolist()

failed_days: list[dict[str, str]] = []
write_chunk_idx = 0

with pg_engine.connect() as conn:
    total_days = len(trade_dates)
    status_display = display("", display_id = True)
    for idx, day in enumerate(tqdm(trade_dates, desc = "days"), start = 1):
        status_display.update(f"處理日期: {pd.to_datetime(day).date()} ({idx}/{total_days})")
        params = {"trade_date": day, "start_time": start_time}
        if end_time:
            params["end_time"] = end_time
        try:
            g = pd.read_sql_query(day_query, conn, params = params)
            if g.empty:
                continue

            if set(volume_cols) <= set(g.columns):
                g[volume_cols] = g[volume_cols].apply(pd.to_numeric, errors = "coerce").fillna(0)
            trade_date = pd.to_datetime(day)
            g["transaction_time"] = pd.to_datetime(
                g["trade_date"].astype(str) + " " + g["transaction_time"].astype(str),
                format="ISO8601",
            )
            close_ts = trade_date + pd.Timedelta(hours = 13, minutes = 30)
            cutoff_ts = trade_date + pd.Timedelta(hours = 13, minutes = 24)

            g["sum_io_rolling_10"] = g["in_out"].rolling(10).sum()
            g["mid_price"] = (g["bid_1_price"] + g["ask_1_price"]) / 2
            g["mid_price_delta"] = g["mid_price"].diff()
            g["spread"] = g["ask_1_price"] - g["bid_1_price"]
            g["spread_delta"] = g["spread"].diff()
            g["price_delta"] = g["price"].ffill().diff()
            g["sum_price_delta_rolling_10"] = g["price_delta"].rolling(10).sum()
            g["momentum"] = g["price"].rolling(100).apply(lambda x: x.iloc[-1] - x.iloc[0])
            g["volatility_10"] = g["price"].rolling(100).std()
            g["hl_range"] = g["price"].rolling(70).max() - g["price"].rolling(10).min()
            g["imbalance"] = (g["bid_1_volume"] - g["ask_1_volume"]) / (g["bid_1_volume"] + g["ask_1_volume"])
            g["delta_bid_1_volume"] = g["bid_1_volume"].diff()
            g["delta_ask_1_volume"] = g["ask_1_volume"].diff()
            g["vol_diff"] = g["volume"].diff()
            g["vol_ma_10"] = g["volume"].rolling(100).mean()
            g["time_to_close"] = (close_ts - g["transaction_time"]).dt.total_seconds()
            g["time_since_prev_trade"] = (
                g["transaction_time"] - g["transaction_time"].where(g["declaration_no"] > 0).shift(1).ffill()
            ).dt.total_seconds()
            g["trade_intensity"] = ((g["declaration_no"] > 0).astype(int)).rolling(10, min_periods=1).sum()
            g["is_trade"] = (g["declaration_no"] > 0).astype(int)
            g["imb_spread"] = g["imbalance"] * g["spread"]
            g["ret_vol"] = g["price_delta"] * g["volatility_10"]
            g["vol_imb"] = g["volume"] * g["imbalance"]
            g["ask_1_up"] = g["ask_1_price"] > g["ask_1_price"].shift(1)
            g["bid_1_down"] = g["bid_1_price"] < g["bid_1_price"].shift(1)

            hist = g[g["transaction_time"] <= cutoff_ts].copy()
            if hist.empty:
                continue

            # 取最後一筆非 0 的價格作為基準與收盤
            hist_nonzero = hist.loc[hist["price"] != 0]
            close_nonzero = g.loc[g["price"] != 0]
            if hist_nonzero.empty or close_nonzero.empty:
                continue

            baseline = hist_nonzero["price"].iloc[-1]
            close_price = close_nonzero["price"].iloc[-1]
            target_label = 1 if close_price > baseline else (-1 if close_price < baseline else 0)
            hist["relative_rise_fall"] = target_label

            hist = hist.fillna(0)
            with pg_engine.begin() as writer:
                hist.to_sql(
                    "t2317_features",
                    con = writer,
                    schema = "public",
                    if_exists = "replace" if write_chunk_idx == 0 else "append",
                    index = False,
                    method = "multi",
                )
            write_chunk_idx += 1
        except Exception as exc:
            failed_days.append({"trade_date": str(pd.to_datetime(day).date()), "error": str(exc)})
            print(f"[ERROR] {pd.to_datetime(day).date()} 失敗: {exc}")
        finally:
            del g
            gc.collect()
'處理日期: 2025-11-16 (212/212)'
days:   0%|          | 0/212 [00:00<?, ?it/s]
/var/folders/np/82kc955j7g9_6x0dw88zjzw40000gn/T/ipykernel_48089/4230218080.py:110: FutureWarning:

Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`

/var/folders/np/82kc955j7g9_6x0dw88zjzw40000gn/T/ipykernel_48089/4230218080.py:110: FutureWarning:

Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`

/var/folders/np/82kc955j7g9_6x0dw88zjzw40000gn/T/ipykernel_48089/4230218080.py:110: FutureWarning:

Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`

/var/folders/np/82kc955j7g9_6x0dw88zjzw40000gn/T/ipykernel_48089/4230218080.py:110: FutureWarning:

Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`

tTX
# tTXF 加到 spot
spot_uri = "postgresql+psycopg2://devuser:DevPass123!@localhost:5432/spot"
future_db = "future"
future_user = "devuser"
future_pass = "DevPass123!"
future_host = "localhost"
future_port = "5432"

target_schema = "public"
target_table = "ms50_2330"
target_qualified = f"{target_schema}.{target_table}"
fdw_schema = "future_fdw"
source_qualified = f'{fdw_schema}."tTXF"'

bucket_ms   = 50
chunk_days  = 30
start_date  = "2025-01-01"
end_date    = "2025-11-30"
start_time  = "09:00:00"
end_time    = "13:30:00"

engine = create_engine(spot_uri)

# 建 FDW 與 foreign table（若已存在會跳過）
with engine.begin() as conn:
    conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS postgres_fdw;")
    conn.exec_driver_sql(f"""
        CREATE SERVER IF NOT EXISTS future_srv
          FOREIGN DATA WRAPPER postgres_fdw
          OPTIONS (host '{future_host}', port '{future_port}', dbname '{future_db}');
    """)
    conn.exec_driver_sql(f"""
        CREATE USER MAPPING IF NOT EXISTS FOR {future_user}
          SERVER future_srv
          OPTIONS (user '{future_user}', password '{future_pass}');
    """)
    conn.exec_driver_sql(f"CREATE SCHEMA IF NOT EXISTS {fdw_schema};")
    conn.exec_driver_sql(f"""
        DROP FOREIGN TABLE IF EXISTS future_fdw."tTXF";
        IMPORT FOREIGN SCHEMA public
            LIMIT TO ("tTXF")
            FROM SERVER future_srv INTO {fdw_schema};
    """)

# 確保欄位存在
cols_to_add = [
    ("txf_price_delta_1s",  "double precision"),
    ("txf_price_delta_5s",  "double precision"),
    ("txf_volume_delta_1s", "bigint"),
    ("txf_volume_delta_5s", "bigint"),
]
with engine.begin() as conn:
    for col, dtype in cols_to_add:
        conn.exec_driver_sql(
            f'ALTER TABLE {target_qualified} ADD COLUMN IF NOT EXISTS {col} {dtype}'
        )

def daterange_chunks(s, e, days):
    cur = datetime.strptime(s, "%Y-%m-%d").date()
    end = datetime.strptime(e, "%Y-%m-%d").date()
    while cur <= end:
        nxt = min(cur + timedelta(days=days-1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)

single_sql = text(f"""
WITH day_list AS (
    SELECT generate_series(
        CAST(:s AS date),
        CAST(:e AS date),
        INTERVAL '1 day'
    ) AS dd
),
source_raw AS (
    SELECT
        dd,
        tt,
        f04 AS price,
        f05 AS volume,
        (
            dd::timestamp
            + (
                floor(EXTRACT(EPOCH FROM tt::time) * 1000 / :bucket_ms)::bigint
                * :bucket_ms * INTERVAL '1 millisecond'
              )
        ) AS bucket_start
    FROM {source_qualified}
    WHERE dd BETWEEN :s AND :e
      AND tt::time >= CAST(:start_time AS time)
      AND (:end_time IS NULL OR tt::time <= CAST(:end_time AS time))
),
bucket_series AS (
    SELECT
        dl.dd,
        generate_series(
            dl.dd::timestamp + CAST(:start_time AS time),
            dl.dd::timestamp + CAST(:end_time   AS time),
            (:bucket_ms)::int * INTERVAL '1 millisecond'
        ) AS bucket_start
    FROM day_list dl
),
agg AS (
    SELECT
        dd,
        bucket_start,
        (array_agg(price ORDER BY tt))[array_length(array_agg(price ORDER BY tt), 1)] AS last_price,
        SUM(volume) AS volume
    FROM source_raw
    GROUP BY dd, bucket_start
),
aligned AS (
    SELECT
        bs.dd,
        bs.bucket_start,
        a.last_price,
        COALESCE(a.volume, 0) AS volume,
        SUM(CASE WHEN a.last_price IS NOT NULL THEN 1 ELSE 0 END)
            OVER (PARTITION BY bs.dd ORDER BY bs.bucket_start) AS grp
    FROM bucket_series bs
    LEFT JOIN agg a
      ON a.dd = bs.dd AND a.bucket_start = bs.bucket_start
),
locf AS (
    SELECT
        dd,
        bucket_start,
        FIRST_VALUE(last_price) OVER (PARTITION BY dd, grp ORDER BY bucket_start) AS ff_price,
        volume
    FROM aligned
),
final AS (
    SELECT
        dd,
        bucket_start,
        COALESCE(ff_price - LAG(ff_price, 20)  OVER w, 0) AS txf_price_delta_1s,
        COALESCE(ff_price - LAG(ff_price, 100) OVER w, 0) AS txf_price_delta_5s,
        COALESCE(volume    - LAG(volume,    20)  OVER w, 0)::bigint AS txf_volume_delta_1s,
        COALESCE(volume    - LAG(volume,    100) OVER w, 0)::bigint AS txf_volume_delta_5s
    FROM locf
    WINDOW w AS (PARTITION BY dd ORDER BY bucket_start)
)
UPDATE {target_qualified} t
SET
    txf_price_delta_1s  = f.txf_price_delta_1s,
    txf_price_delta_5s  = f.txf_price_delta_5s,
    txf_volume_delta_1s = f.txf_volume_delta_1s,
    txf_volume_delta_5s = f.txf_volume_delta_5s
FROM final f
WHERE t.trade_date   = f.dd
  AND t.bucket_start = f.bucket_start;
""")

for chunk_start, chunk_end in tqdm(list(daterange_chunks(start_date, end_date, chunk_days)),
                                   desc="txf deltas via FDW (single SQL)"):
    params = {
        "s": chunk_start,
        "e": chunk_end,
        "start_time": start_time,
        "end_time": end_time,
        "bucket_ms": bucket_ms,
    }
    with engine.begin() as conn:
        conn.execute(single_sql, params)

# 保底補 0
with engine.begin() as conn:
    conn.exec_driver_sql(f"""
        UPDATE {target_qualified}
        SET
            txf_price_delta_1s  = COALESCE(txf_price_delta_1s,  0),
            txf_price_delta_5s  = COALESCE(txf_price_delta_5s,  0),
            txf_volume_delta_1s = COALESCE(txf_volume_delta_1s, 0),
            txf_volume_delta_5s = COALESCE(txf_volume_delta_5s, 0)
        WHERE txf_price_delta_1s  IS NULL
           OR txf_price_delta_5s  IS NULL
           OR txf_volume_delta_1s IS NULL
           OR txf_volume_delta_5s IS NULL;
    """)

print("Done: FDW 單一 SQL 更新 txf price/volume 1s/5s 差分。")
txf deltas via FDW (single SQL):   0%|          | 0/12 [00:00<?, ?it/s]
Done: FDW 單一 SQL 更新 txf price/volume 1s/5s 差分。
LSTM
preprocess
func
# 封裝三分類 LSTM 訓練流程成函式，外部自帶 train/test 資料
# 傳入: train_X (N, T, F), train_y (N,), test_X, test_y
# label 需為 0/1/2；若不用驗證，把 use_val=False 即可
def train_price_lstm_cls3(
    train_X,
    train_y,
    test_X,
    test_y,
    *,
    use_val: bool = True,
    val_ratio: float = 0.1,
    early_stop: bool = True,
    patience: int = 5,
    min_delta: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 30,
    lr: float = 1e-3,
    hidden: int = 128,
    layers: int = 2,
    dropout: float = 0.2,
    seed: int = 42,
    scale: bool = True,
    device=None,
) -> Tuple[nn.Module, Dict[str, float], pd.DataFrame]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if device is None:
        use_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_built() and torch.backends.mps.is_available()
        device = torch.device("mps" if use_mps else "cpu")

    # 縮放特徵
    scaler = StandardScaler() if scale else None
    if scale:
        train_X = scaler.fit_transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
        test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    train_tensor = torch.tensor(train_X, dtype = torch.float32)
    train_targets = torch.tensor(train_y, dtype = torch.long)
    test_tensor = torch.tensor(test_X, dtype = torch.float32)
    test_targets = torch.tensor(test_y, dtype = torch.long)

    full_train_ds = TensorDataset(train_tensor, train_targets)
    if use_val and len(full_train_ds) > 1:
        val_size = max(1, int(val_ratio * len(full_train_ds)))
        train_size = len(full_train_ds) - val_size
        train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        train_ds = full_train_ds
        val_loader = None
    test_ds = TensorDataset(test_tensor, test_targets)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    class PriceLSTMCls3(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 3),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last)

    model = PriceLSTMCls3(train_X.shape[-1], hidden, layers, dropout).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_val = float('inf')
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            loss = crit(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item() * bx.size(0)
        train_loss = total / len(train_loader.dataset)

        if val_loader:
            model.eval()
            vtotal = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    vtotal += crit(model(bx), by).item() * bx.size(0)
            val_loss = vtotal / len(val_loader.dataset)
        else:
            val_loss = train_loss

        if epoch == 1 or epoch % 5 == 0:
            msg = f"[Cls3] Epoch {epoch:02d} | train = {train_loss:.4f}"
            if val_loader:
                msg += f" | val = {val_loss:.4f}"
            print(msg)

        if val_loss + min_delta < best_val:
            best_val = val_loss
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if val_loader and early_stop and no_improve >= patience:
                print(f"[Cls3] Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # 評估
    model.eval()
    logits_list, targets_list = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits = model(bx).cpu().numpy()
            logits_list.append(logits)
            targets_list.append(by.numpy())

    logits_arr = np.concatenate(logits_list)
    targets_arr = np.concatenate(targets_list)
    probs = np.exp(logits_arr - logits_arr.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)

    unique, counts = np.unique(preds, return_counts=True)
    print("Pred class distribution:", dict(zip(unique, counts)))


    acc = accuracy_score(targets_arr, preds)
    report = classification_report(targets_arr, preds, digits=3, zero_division=0, output_dict=True)
    metrics = {"accuracy": acc, "report": report}

    pred_df = pd.DataFrame(
        {
            "actual_cls": targets_arr.astype(int),
            "pred_cls": preds,
            "prob_down": probs[:, 0],
            "prob_flat": probs[:, 1],
            "prob_up": probs[:, 2],
        }
    )
    return model, metrics, pred_df

# 使用範例（假設你已準備好 numpy array）：
# model, metrics, preds = train_price_lstm_cls3(train_X, train_y, test_X, test_y, use_val=True, early_stop=True)
# print(metrics["accuracy"])
# display(pd.DataFrame(metrics["report"]).T)
# display(preds.head())
def compute_lstm_shap(model, train_tensor, test_tensor, device, feature_names, max_background=20, max_samples=10):
    if train_tensor is None or test_tensor is None:
        raise RuntimeError('請先執行 LSTM cell 產生 train/test tensor 再計算 SHAP。')
    if len(train_tensor) == 0 or len(test_tensor) == 0:
        raise RuntimeError('SHAP 需要一些 train/test 樣本，請先完成資料切分與訓練。')

    model.eval()
    background_size = min(max_background, len(train_tensor))
    sample_size = min(max_samples, len(test_tensor))
    if background_size == 0 or sample_size == 0:
        raise RuntimeError('SHAP 需要一些 train/test 樣本，請先完成資料切分與訓練。')

    background_data = train_tensor[:background_size].to(device)
    sample_data = test_tensor[:sample_size].to(device)

    explainer = shap.GradientExplainer(model, background_data)
    shap_values = explainer.shap_values(sample_data)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    vals = np.abs(np.asarray(shap_values))
    if vals.ndim == 4:  # (samples, seq_len, features, classes)
        importance = vals.mean(axis=(0, 1, 3))
    elif vals.ndim == 3:  # (samples, seq_len, features)
        importance = vals.mean(axis=(0, 1))
    elif vals.ndim == 2:
        importance = vals.mean(axis=0)
    else:
        raise ValueError(f'Unexpected SHAP shape: {vals.shape}')

    return (
        pd.Series(importance, index=feature_names)
        .sort_values(ascending=False)
    )
read
# ---- DB / table config ----
pg_uri = "postgresql+psycopg2://devuser:DevPass123!@localhost:5432/spot"
feature_schema = "public"
feature_table = "t2330_features_final"  # 可改成 ms50_2330 等
feature_time_col = "transaction_time"  # 若是 bucket 表請改成 "bucket_start"
feature_qualified = f"{feature_schema}.{feature_table}"
pg_engine = create_engine(pg_uri, future=True)

# ---- time/date window ----
start_time = "13:20:00"
end_time = "13:25:00"
start_date_override: str | None = "2025-01-01"
end_date_override: str | None = "2025-11-30"

time_filter = f"{feature_time_col}::time BETWEEN :start_time AND :end_time" if end_time else f"{feature_time_col}::time >= :start_time"

slice_query = text(f"""
SELECT *
FROM {feature_qualified}
WHERE {time_filter}
  AND trade_date BETWEEN COALESCE(:start_date, trade_date) AND COALESCE(:end_date, trade_date)
ORDER BY trade_date, {feature_time_col}
""")

params = {
    "start_time": start_time,
    "start_date": start_date_override,
    "end_date": end_date_override,
}
if end_time:
    params["end_time"] = end_time

with pg_engine.connect() as conn:
    df = pd.read_sql_query(slice_query, conn, params=params)

df = df.sort_values(["trade_date", feature_time_col]).reset_index(drop = True)
df.dropna(inplace = True)


[SQL: 
SELECT *
FROM public.t2330_features_final
WHERE transaction_time::time BETWEEN %(start_time)s AND %(end_time)s
  AND trade_date BETWEEN COALESCE(%(start_date)s, trade_date) AND COALESCE(%(end_date)s, trade_date)
ORDER BY trade_date, transaction_time
]
[parameters: {'start_time': '13:20:00', 'end_time': '13:25:00', 'start_date': '2025-01-01', 'end_date': '2025-11-30'}]
(Background on this error at: https://sqlalche.me/e/20/f405)
predict close
event base
general
SEQ_LEN_PRICE = 1000
BATCH_PRICE = 64
EPOCHS_PRICE = 30
LR_PRICE = 1e-3
HIDDEN_PRICE = 128
LAYERS_PRICE = 2
DROPOUT_PRICE = 0.2
PATIENCE_PRICE = 5
MIN_DELTA_PRICE = 1e-4
USE_VAL_PRICE = True
EARLY_STOP_PRICE = False
RANDOM_SEED = 42
SCALE = True
sequences, targets, dates = [], [], []

event_features = [
    # ----- Price & Returns -----
    "last_trade_price",     # LOCF 後的實際交易價格
    "price_delta",          # event-to-event Δprice

    # ----- Order-flow -----
    "in_out",               # 單筆買賣壓（方向 + 大小）
    "in_out_rolling_5",     # 過去 5 events 累積 order flow
    "in_out_rolling_20",    # 過去 20 events 累積 order flow

    # ----- Volume (event-based) -----
    "trade_volume",         # 單筆成交量
    "trade_vol_20ev",       # 過去 20 events 累積成交量
    "trade_vol_100ev",      # 過去 100 events 累積成交量

    # ----- Delta Depth (L1/L2 變化) -----
    "delta_bid_1_volume",   # L1 買量變化
    "delta_ask_1_volume",   # L1 賣量變化
    "delta_bid_2_volume",   # L2 買量變化
    "delta_ask_2_volume",   # L2 賣量變化

    # ----- Microstructure Prices -----
    "mid_price",            # (bid1 + ask1) / 2
    "spread",               # ask1 - bid1

    # ----- Event-based Volatility -----
    "vol_20ev",             # 過去 20 events 標準差
    "vol_100ev",            # 過去 100 events 標準差

    # ----- Event-based Order-imbalance -----
    "io_imb_20ev",          # 過去 20 events 買賣壓比率
    "io_imb_100ev",         # 過去 100 events 買賣壓比率

    # ----- Event-based Order-imbalance -----
    "txf_price_delta_20ev",          # 過去 20 events 價格變動
    "txf_price_delta_100ev",         # 過去 100 events 價格變動
    "txf_volume_delta_20ev",         # 過去 20 events 成交量變動
    "txf_volume_delta_100ev",        # 過去 100 events 成交量變動
]

for d, g in features.groupby("trade_date", sort=True):
    g = g.copy()
    g.loc[:, price_features] = g[price_features].ffill()

    label_raw = g["relative_rise_fall"].iloc[-1]
    if label_raw not in (-1, 0, 1):
        continue
    label = int(label_raw + 1)  # -1/0/1 -> 0/1/2

    cutoff = pd.Timestamp(d) + pd.Timedelta(hours=13, minutes=24)
    cut = g[g["transaction_time"] <= cutoff]
    if len(cut) < SEQ_LEN_PRICE:
        continue

    window = cut.iloc[-SEQ_LEN_PRICE:][price_features].to_numpy()
    sequences.append(window)
    targets.append(label)
    dates.append(d)

sequences = np.array(sequences)
targets = np.array(targets)

# 切 train/test
split_idx = max(int(len(sequences) * 0.8), 1)
train_Xp, test_Xp = sequences[:split_idx], sequences[split_idx:]
train_yp, test_yp = targets[:split_idx], targets[split_idx:]

model, metrics, pred_df = train_price_lstm_cls3(
    train_Xp, train_yp, test_Xp, test_yp,
    use_val = USE_VAL_PRICE,
    early_stop = EARLY_STOP_PRICE,
    patience = PATIENCE_PRICE,
    min_delta = MIN_DELTA_PRICE,
    batch_size = BATCH_PRICE,
    epochs = EPOCHS_PRICE,
    lr = LR_PRICE,
    hidden = HIDDEN_PRICE,
    layers = LAYERS_PRICE,
    dropout = DROPOUT_PRICE,
    seed = RANDOM_SEED,
    scale = SCALE
)

display(pd.DataFrame(metrics["report"]).T)
display(pred_df)
[Cls3] Epoch 01 | train = 1.0930 | val = 1.0663
[Cls3] Epoch 05 | train = 0.9987 | val = 1.0971
[Cls3] Epoch 10 | train = 0.8782 | val = 1.0909
[Cls3] Epoch 15 | train = 0.7524 | val = 1.1123
[Cls3] Epoch 20 | train = 0.5900 | val = 1.1230
[Cls3] Epoch 25 | train = 0.4199 | val = 1.1974
[Cls3] Epoch 30 | train = 0.3115 | val = 1.2604
Pred class distribution: {np.int64(2): np.int64(40)}
precision	recall	f1-score	support
0	0.000000	0.000000	0.000000	19.000
1	0.000000	0.000000	0.000000	12.000
2	0.225000	1.000000	0.367347	9.000
accuracy	0.225000	0.225000	0.225000	0.225
macro avg	0.075000	0.333333	0.122449	40.000
weighted avg	0.050625	0.225000	0.082653	40.000
actual_cls	pred_cls	prob_down	prob_flat	prob_up
0	0	2	0.288848	0.341909	0.369243
1	1	2	0.276202	0.352370	0.371428
2	0	2	0.279602	0.348176	0.372222
3	1	2	0.286705	0.344724	0.368571
4	2	2	0.278739	0.348492	0.372769
5	1	2	0.280728	0.348493	0.370779
6	2	2	0.290007	0.343156	0.366837
7	1	2	0.294332	0.338753	0.366915
8	0	2	0.272004	0.360744	0.367252
9	1	2	0.286832	0.343167	0.370002
10	2	2	0.277005	0.351184	0.371812
11	2	2	0.275201	0.355252	0.369548
12	0	2	0.278732	0.349570	0.371698
13	0	2	0.288731	0.340316	0.370953
14	0	2	0.289989	0.342620	0.367391
15	0	2	0.284396	0.345470	0.370134
16	2	2	0.285824	0.344675	0.369501
17	0	2	0.291675	0.340912	0.367413
18	1	2	0.285701	0.345535	0.368764
19	1	2	0.273820	0.354126	0.372055
20	1	2	0.292154	0.341440	0.366406
21	2	2	0.284994	0.345212	0.369794
22	2	2	0.280482	0.349010	0.370508
23	0	2	0.282858	0.346687	0.370455
24	0	2	0.285638	0.345346	0.369016
25	1	2	0.286881	0.344986	0.368134
26	0	2	0.289336	0.342680	0.367984
27	0	2	0.277934	0.350270	0.371796
28	1	2	0.290009	0.342778	0.367213
29	1	2	0.285961	0.347347	0.366691
30	0	2	0.288825	0.343504	0.367672
31	1	2	0.290814	0.341720	0.367466
32	0	2	0.280377	0.347829	0.371794
33	0	2	0.284826	0.346600	0.368574
34	2	2	0.288702	0.343654	0.367644
35	2	2	0.290027	0.342935	0.367038
36	0	2	0.287938	0.344412	0.367650
37	0	2	0.286725	0.343979	0.369296
38	0	2	0.280689	0.347740	0.371572
39	0	2	0.286633	0.344354	0.369013
# 針對上一個 cell 的價格 LSTM，建立 SHAP 所需的 tensor
n_features_price = len(price_features)
if SCALE:
    price_scaler = StandardScaler()
    train_Xp_scaled = price_scaler.fit_transform(train_Xp.reshape(-1, n_features_price)).reshape(train_Xp.shape)
    test_Xp_scaled = price_scaler.transform(test_Xp.reshape(-1, n_features_price)).reshape(test_Xp.shape)
else:
    train_Xp_scaled = train_Xp.copy()
    test_Xp_scaled = test_Xp.copy()

price_train_tensor = torch.tensor(train_Xp_scaled, dtype = torch.float32)
price_test_tensor = torch.tensor(test_Xp_scaled, dtype = torch.float32)
price_device = next(model.parameters()).device

price_shap_importance = compute_lstm_shap(
    model = model,
    train_tensor = price_train_tensor,
    test_tensor = price_test_tensor,
    device = price_device,
    feature_names = price_features
)
display(price_shap_importance.head(10).to_frame(name = 'mean|SHAP|'))
mean|SHAP|
imbalance	0.000006
imb_spread	0.000004
vol_imb	0.000003
price	0.000003
trade_intensity	0.000003
volatility_10	0.000003
momentum	0.000002
mid_price	0.000002
is_trade	0.000002
time_since_prev_trade	0.000002
block
SEQ_LEN_PRICE = 1000
BATCH_PRICE = 64
EPOCHS_PRICE = 30
LR_PRICE = 1e-3
HIDDEN_PRICE = 128
LAYERS_PRICE = 2
DROPOUT_PRICE = 0.2
PATIENCE_PRICE = 5
MIN_DELTA_PRICE = 1e-4
USE_VAL_PRICE = True
EARLY_STOP_PRICE = False
RANDOM_SEED = 42
SCALE = True
sequences, targets, dates = [], [], []

price_features = [
    "in_out", "price", "mid_price", "mid_price_delta", "spread",
    "spread_delta", "price_delta", "momentum", "volatility_10",
    "hl_range", "imbalance", "delta_bid_1_volume", "delta_ask_1_volume",
    "vol_diff", "vol_ma_10", "time_to_close", "time_since_prev_trade",
    "trade_intensity", "imb_spread", "ret_vol", "vol_imb", "is_trade",
]

for d, g in features.groupby("trade_date", sort = True):
    g = g.copy()
    g.loc[:, price_features] = g[price_features].ffill()

    label_raw = g["relative_rise_fall"].iloc[-1]
    if label_raw not in (-1, 0, 1):
        continue
    label = int(label_raw + 1)  # -1/0/1 -> 0/1/2

    cutoff = pd.Timestamp(d) + pd.Timedelta(hours=13, minutes=24)
    cut = g[g["transaction_time"] <= cutoff]
    if len(cut) < SEQ_LEN_PRICE:
        continue

    window = cut.iloc[-SEQ_LEN_PRICE:][price_features].to_numpy()
    sequences.append(window)
    targets.append(label)
    dates.append(d)

sequences = np.array(sequences)
targets = np.array(targets)

block_size = 50

def block_pool(x, block):
    usable = (x.shape[1] // block) * block  # 只取可整除的部分
    x_cut = x[:, :usable, :]
    x_view = x_cut.reshape(x.shape[0], -1, block, x.shape[2])
    return x_view.mean(axis = 2)

# 切 train/test，並在套用 block pooling 後仍沿用 train_Xp/test_Xp 命名，方便後續流程和 SHAP
split_idx = max(int(len(sequences) * 0.8), 1)
train_Xp_raw, test_Xp_raw = sequences[:split_idx], sequences[split_idx:]
train_yp, test_yp = targets[:split_idx], targets[split_idx:]

train_Xp = block_pool(train_Xp_raw, block_size)
test_Xp  = block_pool(test_Xp_raw,  block_size)

model, metrics, pred_df = train_price_lstm_cls3(
    train_Xp, train_yp, test_Xp, test_yp,
    use_val = USE_VAL_PRICE,
    early_stop = EARLY_STOP_PRICE,
    patience = PATIENCE_PRICE,
    min_delta = MIN_DELTA_PRICE,
    batch_size = BATCH_PRICE,
    epochs = EPOCHS_PRICE,
    lr = LR_PRICE,
    hidden = HIDDEN_PRICE,
    layers = LAYERS_PRICE,
    dropout = DROPOUT_PRICE,
    seed = RANDOM_SEED,
    scale = SCALE
)

display(pd.DataFrame(metrics["report"]).T)
display(pred_df)
[Cls3] Epoch 01 | train = 1.0929 | val = 1.0658
[Cls3] Epoch 05 | train = 1.0093 | val = 1.0815
[Cls3] Epoch 10 | train = 0.8919 | val = 1.0607
[Cls3] Epoch 15 | train = 0.7568 | val = 1.0526
[Cls3] Epoch 20 | train = 0.5929 | val = 1.0885
[Cls3] Epoch 25 | train = 0.4049 | val = 1.1380
[Cls3] Epoch 30 | train = 0.2611 | val = 1.1551
Pred class distribution: {np.int64(0): np.int64(2), np.int64(1): np.int64(10), np.int64(2): np.int64(28)}
precision	recall	f1-score	support
0	0.000000	0.000000	0.000000	19.0
1	0.300000	0.250000	0.272727	12.0
2	0.178571	0.555556	0.270270	9.0
accuracy	0.200000	0.200000	0.200000	0.2
macro avg	0.159524	0.268519	0.180999	40.0
weighted avg	0.130179	0.200000	0.142629	40.0
actual_cls	pred_cls	prob_down	prob_flat	prob_up
0	0	2	0.310221	0.172896	0.516882
1	1	2	0.220173	0.267102	0.512725
2	0	1	0.207380	0.427756	0.364864
3	1	2	0.367193	0.189207	0.443600
4	2	2	0.265062	0.269300	0.465638
5	1	2	0.231784	0.284289	0.483927
6	2	0	0.349721	0.330123	0.320157
7	1	2	0.363891	0.241710	0.394399
8	0	1	0.253639	0.440094	0.306267
9	1	2	0.281457	0.262472	0.456072
10	2	1	0.253163	0.377412	0.369424
11	2	2	0.350893	0.244050	0.405057
12	0	1	0.260107	0.388093	0.351800
13	0	2	0.250369	0.293897	0.455734
14	0	2	0.355981	0.283365	0.360653
15	0	1	0.180796	0.481308	0.337896
16	2	2	0.222183	0.369312	0.408505
17	0	2	0.196560	0.386974	0.416466
18	1	2	0.204978	0.385427	0.409595
19	1	1	0.272862	0.393432	0.333706
20	1	2	0.211126	0.345008	0.443866
21	2	2	0.216382	0.389929	0.393689
22	2	1	0.224806	0.420234	0.354960
23	0	1	0.183422	0.488953	0.327626
24	0	2	0.237939	0.323929	0.438132
25	1	1	0.217000	0.396375	0.386625
26	0	2	0.309740	0.235075	0.455185
27	0	2	0.320843	0.285179	0.393978
28	1	2	0.314644	0.201257	0.484099
29	1	1	0.286157	0.380524	0.333318
30	0	2	0.250005	0.287021	0.462974
31	1	2	0.282970	0.208745	0.508285
32	0	2	0.249549	0.331323	0.419128
33	0	2	0.305663	0.234028	0.460309
34	2	2	0.274251	0.241645	0.484104
35	2	0	0.366533	0.271064	0.362403
36	0	2	0.246329	0.289657	0.464014
37	0	2	0.295122	0.275294	0.429585
38	0	2	0.335966	0.283976	0.380057
39	0	2	0.269540	0.290864	0.439596
n_features_price = len(price_features)
if SCALE:
    block_scaler = StandardScaler()
    train_Xp_scaled = block_scaler.fit_transform(train_Xp.reshape(-1, n_features_price)).reshape(train_Xp.shape)
    test_Xp_scaled = block_scaler.transform(test_Xp.reshape(-1, n_features_price)).reshape(test_Xp.shape)
else:
    train_Xp_scaled = train_Xp.copy()
    test_Xp_scaled = test_Xp.copy()

block_train_tensor = torch.tensor(train_Xp_scaled, dtype = torch.float32)
block_test_tensor = torch.tensor(test_Xp_scaled, dtype = torch.float32)
block_device = next(model.parameters()).device

block_shap_importance = compute_lstm_shap(
    model = model,
    train_tensor = block_train_tensor,
    test_tensor = block_test_tensor,
    device = block_device,
    feature_names = price_features,
    max_background = 20,
    max_samples = 10
)
display(block_shap_importance.head(10).to_frame(name = 'mean|SHAP|'))
mean|SHAP|
hl_range	0.005920
vol_diff	0.003953
is_trade	0.003565
imb_spread	0.003403
time_since_prev_trade	0.003034
volatility_10	0.002866
vol_ma_10	0.002814
ret_vol	0.002472
price	0.002453
vol_imb	0.002441
time base
relative rise fall
general
SAMPLE_START_TIME = "13:20:00"   # 樣本起始時間（含）
SAMPLE_END_TIME   = "13:24:00"   # 樣本結束時間（含）
SEQ_LEN_OVERRIDE  = None          # 如需固定序列長度（行數），填整數；None 則用時間窗內的行數

start_t = datetime.strptime(SAMPLE_START_TIME, "%H:%M:%S").time()
end_t   = datetime.strptime(SAMPLE_END_TIME, "%H:%M:%S").time()

BATCH = 32
EPOCHS = 30
LR = 1e-3
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.2
PATIENCE = 5
MIN_DELTA = 1e-4
USE_VAL = True
EARLY_STOP = False
RANDOM_SEED = 42
SCALE = True
sequences, targets, dates = [], [], []

price_features = [
    # --- Price-based ---
    "last_price",
    "price_delta",
    "return_1s",
    "return_5s",

    # --- Time features ---
    "time_to_close_s",

    # --- Volume / Intensity ---
    "volume",
    "vol_1s",
    "vol_5s",
    "volume_1s",
    "volume_5s",

    # --- Order-flow imbalance ---
    "io_imb_1s",
    "io_imb_5s",

    # --- Level 1/2 book info ---
    "bid_1_price",
    "ask_1_price",
    "bid_1_volume",
    "ask_1_volume",
    "bid_1_volume_delta",
    "ask_1_volume_delta",

    "bid_2_price",
    "bid_2_volume",
    "ask_2_price",
    "ask_2_volume",

    # --- Book imbalance ---
    "depth_imb",

    # --- TXF features ---
    "txf_price_delta_1s",
    "txf_price_delta_5s",
    "txf_volume_delta_1s",
    "txf_volume_delta_5s",
]

for d, g in df.groupby("trade_date", sort=True):
    g = g.copy()
    g.loc[:, price_features] = g[price_features].ffill()
    # 確保 bucket_start 為 datetime 型態
    if not np.issubdtype(g["bucket_start"].dtype, np.datetime64):
        g["bucket_start"] = pd.to_datetime(g["bucket_start"])

    label_raw = g["relative_rise_fall"].iloc[-1]
    if label_raw not in (-1, 0, 1):
        continue
    label = int(label_raw + 1)  # -1/0/1 -> 0/1/2

    bucket_time = g["bucket_start"].dt.time
    mask = (bucket_time >= start_t) & (bucket_time <= end_t)
    cut = g[mask]
    if cut.empty:
        continue

    seq_len = int(SEQ_LEN_OVERRIDE) if SEQ_LEN_OVERRIDE else len(cut)
    if len(cut) < seq_len:
        continue

    window = cut.iloc[-seq_len:][price_features].to_numpy()
    sequences.append(window)
    targets.append(label)
    dates.append(d)

sequences = np.array(sequences)
targets = np.array(targets)

# 切 train/test，避免 test 為空
split_idx = max(int(len(sequences) * 0.8), 1)
train_Xp, test_Xp = sequences[:split_idx], sequences[split_idx:]
train_yp, test_yp = targets[:split_idx], targets[split_idx:]
if len(test_Xp) == 0:
    test_Xp, test_yp = train_Xp, train_yp

model, metrics, pred_df = train_price_lstm_cls3(
    train_Xp, train_yp, test_Xp, test_yp,
    use_val = USE_VAL,
    early_stop = EARLY_STOP,
    patience = PATIENCE,
    min_delta = MIN_DELTA,
    batch_size = BATCH,
    epochs = EPOCHS,
    lr = LR,
    hidden = HIDDEN,
    layers = LAYERS,
    dropout = DROPOUT,
    seed = RANDOM_SEED,
    scale = SCALE,
)

display(pd.DataFrame(metrics["report"]).T)
display(pred_df)
[Cls3] Epoch 01 | train = 1.0977 | val = 1.0958
[Cls3] Epoch 05 | train = 0.9211 | val = 1.0757
[Cls3] Epoch 10 | train = 0.6786 | val = 1.1317
[Cls3] Epoch 15 | train = 0.4879 | val = 1.3491
[Cls3] Epoch 20 | train = 0.3870 | val = 1.5349
[Cls3] Epoch 25 | train = 0.2234 | val = 1.8102
[Cls3] Epoch 30 | train = 0.1593 | val = 2.3428
Pred class distribution: {np.int64(0): np.int64(17), np.int64(1): np.int64(23), np.int64(2): np.int64(1)}
precision	recall	f1-score	support
0	0.470588	0.571429	0.516129	14.000000
1	0.521739	0.571429	0.545455	21.000000
2	0.000000	0.000000	0.000000	6.000000
accuracy	0.487805	0.487805	0.487805	0.487805
macro avg	0.330776	0.380952	0.353861	41.000000
weighted avg	0.427921	0.487805	0.455618	41.000000
actual_cls	pred_cls	prob_down	prob_flat	prob_up
0	1	0	0.505146	0.408148	0.086706
1	0	1	0.136224	0.584071	0.279706
2	1	1	0.396257	0.462557	0.141187
3	0	0	0.478672	0.443847	0.077481
4	2	1	0.170175	0.524171	0.305654
5	1	1	0.290995	0.515978	0.193028
6	2	1	0.210225	0.614670	0.175105
7	1	1	0.224786	0.581900	0.193314
8	0	0	0.525947	0.365231	0.108823
9	1	0	0.473228	0.444990	0.081782
10	0	1	0.433391	0.453479	0.113130
11	1	0	0.508808	0.417037	0.074155
12	0	0	0.516189	0.392448	0.091363
13	0	0	0.461611	0.458940	0.079449
14	1	1	0.370902	0.537498	0.091600
15	0	1	0.452254	0.461097	0.086649
16	2	1	0.138727	0.602432	0.258841
17	1	0	0.488985	0.441820	0.069195
18	2	1	0.435943	0.477883	0.086174
19	1	0	0.534741	0.392652	0.072607
20	2	1	0.377888	0.513075	0.109037
21	1	1	0.385811	0.511361	0.102828
22	1	1	0.434649	0.439514	0.125837
23	1	1	0.412309	0.487839	0.099852
24	1	1	0.301805	0.587556	0.110638
25	0	2	0.194341	0.356205	0.449454
26	1	1	0.282055	0.596653	0.121293
27	0	0	0.501903	0.432503	0.065595
28	0	0	0.525478	0.403625	0.070897
29	0	1	0.429256	0.499365	0.071378
30	1	0	0.467103	0.460293	0.072604
31	1	1	0.429720	0.478309	0.091972
32	1	1	0.370692	0.441417	0.187891
33	0	0	0.455374	0.378473	0.166153
34	1	1	0.391830	0.420309	0.187861
35	0	1	0.409713	0.414178	0.176109
36	1	0	0.427127	0.412583	0.160290
37	2	1	0.275659	0.480721	0.243620
38	1	0	0.412075	0.402173	0.185752
39	0	0	0.424748	0.379870	0.195382
40	1	0	0.434097	0.404618	0.161286
total_samples = len(train_Xp) + len(test_Xp)
if total_samples == 0:
    raise ValueError("No samples to explain: train/test 為空")

n_features_price = len(price_features)
if SCALE:
    scaler = StandardScaler()
    train_Xp_scaled = scaler.fit_transform(train_Xp.reshape(-1, n_features_price)).reshape(train_Xp.shape)
    test_Xp_scaled  = scaler.transform(test_Xp.reshape(-1, n_features_price)).reshape(test_Xp.shape)
else:
    train_Xp_scaled = train_Xp.copy()
    test_Xp_scaled  = test_Xp.copy()

train_tensor = torch.tensor(train_Xp_scaled, dtype=torch.float32)
test_tensor  = torch.tensor(test_Xp_scaled, dtype=torch.float32)
device = next(model.parameters()).device

shap_importance = compute_lstm_shap(
    model=model,
    train_tensor=train_tensor,
    test_tensor=test_tensor,
    device=device,
    feature_names=price_features,
    max_background=20,  # 可視算力調整
    max_samples=10,      # 可視算力調整
)

display(shap_importance.head(10).to_frame(name='mean|SHAP|'))
block
# LSTM 訓練（每日一樣本，13:20~13:24，將 4 分鐘的 bucket 分成多個 block，對每 block 取平均形成序列）
SAMPLE_START_TIME = "13:20:00"
BUCKET_MS = 50
BLOCKS_PER_DAY = 20                # 想要的 block 數（4 分鐘 / BLOCKS_PER_DAY）
BUCKETS_PER_BLOCK = int((4 * 60 * 1000 / BUCKET_MS) // BLOCKS_PER_DAY)  # 4 分鐘總 bucket / block 數

BATCH = 32
EPOCHS = 30
LR = 1e-3
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.2
PATIENCE = 5
MIN_DELTA = 1e-4
USE_VAL = True
EARLY_STOP = False
RANDOM_SEED = 42
SCALE = True

start_dt = datetime.strptime(SAMPLE_START_TIME, "%H:%M:%S")
start_t = start_dt.time()
end_exclusive = (start_dt + timedelta(minutes=4)).time()  # 13:24:00，使用 < 排除結束點

sequences, targets, dates = [], [], []
price_features = [
    "last_price", "price_delta", "return_1s", "return_5s",
    "time_to_close_s",
    "volume", "vol_1s", "vol_5s", "volume_1s", "volume_5s",
    "io_imb_1s", "io_imb_5s",
    "bid_1_price", "ask_1_price",
    "bid_1_volume", "ask_1_volume",
    "bid_1_volume_delta", "ask_1_volume_delta",
    "bid_2_price", "bid_2_volume", "ask_2_price", "ask_2_volume",
    "depth_imb",
]

f = df
if not hasattr(f["bucket_start"], 'dt'):
    f["bucket_start"] = pd.to_datetime(f["bucket_start"])
f = f.sort_values(["trade_date", "bucket_start"]).reset_index(drop=True)

for d, g in df.groupby("trade_date", sort=True):
    g = g.copy()
    g.loc[:, price_features] = g[price_features].ffill()
    label_raw = g["relative_rise_fall"].iloc[-1]
    if label_raw not in (-1, 0, 1):
        continue
    label = int(label_raw + 1)

    bt = g["bucket_start"].dt.time
    cut = g[(bt >= start_t) & (bt < end_exclusive)]
    if len(cut) != BUCKETS_PER_BLOCK * BLOCKS_PER_DAY:
        continue

    blocks = []
    for i in range(BLOCKS_PER_DAY):
        start_idx = i * BUCKETS_PER_BLOCK
        end_idx = start_idx + BUCKETS_PER_BLOCK
        block = cut.iloc[start_idx:end_idx][price_features]
        block_mean = block.mean()
        blocks.append(block_mean.to_numpy())

    seq = np.vstack(blocks)  # shape (BLOCKS_PER_DAY, feature_size)
    sequences.append(seq)
    targets.append(label)
    dates.append(d)

sequences = np.array(sequences)  # shape: (num_days, BLOCKS_PER_DAY, feature_size)
targets = np.array(targets)

if len(sequences) == 0:
    raise ValueError("No samples collected: 時間窗內筆數不足或分段不滿 BLOCKS_PER_DAY，請調整條件。")

# 依交易日切 train/test，同一天不拆分
unique_dates = sorted(set(dates))
split_d = max(int(len(unique_dates) * 0.8), 1)
train_dates = set(unique_dates[:split_d])
test_dates = set(unique_dates[split_d:])

train_idx = [i for i, d in enumerate(dates) if d in train_dates]
test_idx  = [i for i, d in enumerate(dates) if d in test_dates]

train_X = sequences[train_idx]
test_X  = sequences[test_idx]
train_y = targets[train_idx]
test_y  = targets[test_idx]
if len(test_X) == 0:
    test_X, test_y = train_X, train_y

model, metrics, pred_df = train_price_lstm_cls3(
    train_X, train_y, test_X, test_y,
    use_val = USE_VAL,
    early_stop = EARLY_STOP,
    patience = PATIENCE,
    min_delta = MIN_DELTA,
    batch_size = BATCH,
    epochs = EPOCHS,
    lr = LR,
    hidden = HIDDEN,
    layers = LAYERS,
    dropout = DROPOUT,
    seed = RANDOM_SEED,
    scale = SCALE,
)

display(pd.DataFrame(metrics["report"]).T)
display(pred_df)
[Cls3] Epoch 01 | train = 1.1070 | val = 1.1013
[Cls3] Epoch 05 | train = 1.0135 | val = 1.1012
[Cls3] Epoch 10 | train = 0.8830 | val = 1.2511
[Cls3] Epoch 15 | train = 0.6461 | val = 1.5299
[Cls3] Epoch 20 | train = 0.3719 | val = 2.2941
[Cls3] Epoch 25 | train = 0.2320 | val = 2.8246
[Cls3] Epoch 30 | train = 0.1656 | val = 3.3334
Pred class distribution: {np.int64(1): np.int64(41)}
precision	recall	f1-score	support
0	0.000000	0.000000	0.000000	14.000000
1	0.512195	1.000000	0.677419	21.000000
2	0.000000	0.000000	0.000000	6.000000
accuracy	0.512195	0.512195	0.512195	0.512195
macro avg	0.170732	0.333333	0.225806	41.000000
weighted avg	0.262344	0.512195	0.346971	41.000000
actual_cls	pred_cls	prob_down	prob_flat	prob_up
0	1	1	0.317938	0.436492	0.245569
1	0	1	0.303497	0.411985	0.284518
2	1	1	0.303752	0.431846	0.264402
3	0	1	0.307675	0.451088	0.241238
4	2	1	0.331548	0.408249	0.260204
5	1	1	0.304275	0.433393	0.262332
6	2	1	0.304815	0.447369	0.247816
7	1	1	0.304453	0.462279	0.233268
8	0	1	0.335868	0.415888	0.248243
9	1	1	0.306685	0.458858	0.234457
10	0	1	0.312584	0.456556	0.230861
11	1	1	0.317780	0.442216	0.240004
12	0	1	0.325538	0.420985	0.253477
13	0	1	0.309501	0.439667	0.250832
14	1	1	0.313383	0.437411	0.249205
15	0	1	0.314346	0.447244	0.238410
16	2	1	0.311573	0.460126	0.228301
17	1	1	0.306795	0.477766	0.215439
18	2	1	0.317983	0.455980	0.226037
19	1	1	0.312117	0.450834	0.237050
20	2	1	0.288883	0.488559	0.222558
21	1	1	0.309868	0.463156	0.226976
22	1	1	0.316977	0.455307	0.227716
23	1	1	0.309204	0.458351	0.232444
24	1	1	0.306379	0.473176	0.220445
25	0	1	0.330922	0.417656	0.251422
26	1	1	0.301526	0.475697	0.222777
27	0	1	0.311277	0.468965	0.219758
28	0	1	0.310886	0.443341	0.245772
29	0	1	0.305875	0.470003	0.224122
30	1	1	0.310667	0.472548	0.216785
31	1	1	0.305912	0.475573	0.218515
32	1	1	0.338872	0.389386	0.271742
33	0	1	0.345547	0.381470	0.272983
34	1	1	0.341224	0.388034	0.270742
35	0	1	0.341457	0.386973	0.271570
36	1	1	0.339789	0.377869	0.282342
37	2	1	0.336322	0.384715	0.278962
38	1	1	0.341179	0.377852	0.280968
39	0	1	0.346455	0.375076	0.278470
40	1	1	0.342200	0.383950	0.273849
n_features_price = len(price_features)
if SCALE:
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X.reshape(-1, n_features_price)).reshape(train_X.shape)
    test_X_scaled  = scaler.transform(test_X.reshape(-1, n_features_price)).reshape(test_X.shape)
else:
    train_X_scaled, test_X_scaled = train_X, test_X

train_t = torch.tensor(train_X_scaled, dtype=torch.float32)
test_t  = torch.tensor(test_X_scaled, dtype=torch.float32)
device  = next(model.parameters()).device

shap_imp = compute_lstm_shap(
    model = model,
    train_tensor = train_t,
    test_tensor = test_t,
    device = device,
    feature_names = price_features,
    max_background = 20,
    max_samples = 10,
)
display(shap_imp.head(10).to_frame(name='mean|SHAP|'))
mean|SHAP|
last_price	0.000545
bid_2_price	0.000520
depth_imb	0.000511
bid_1_volume	0.000250
io_imb_1s	0.000209
bid_1_volume_delta	0.000190
bid_1_price	0.000165
ask_1_price	0.000161
volume	0.000136
ask_2_price	0.000118
relative vwap rise fall
general
SAMPLE_START_TIME = "13:20:00"   # 樣本起始時間（含）
SAMPLE_END_TIME   = "13:24:00"   # 樣本結束時間（含）
SEQ_LEN_OVERRIDE  = None          # 如需固定序列長度（行數），填整數；None 則用時間窗內的行數

start_t = datetime.strptime(SAMPLE_START_TIME, "%H:%M:%S").time()
end_t   = datetime.strptime(SAMPLE_END_TIME, "%H:%M:%S").time()

BATCH = 32
EPOCHS = 30
LR = 1e-3
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.2
PATIENCE = 5
MIN_DELTA = 1e-4
USE_VAL = True
EARLY_STOP = False
RANDOM_SEED = 42
SCALE = True
sequences, targets, dates = [], [], []

price_features = [
    # --- Price-based ---
    "last_price",
    "price_delta",
    "return_1s",
    "return_5s",

    # --- Time features ---
    "time_to_close_s",

    # --- Volume / Intensity ---
    "volume",
    "vol_1s",
    "vol_5s",
    "volume_1s",
    "volume_5s",

    # --- Order-flow imbalance ---
    "io_imb_1s",
    "io_imb_5s",

    # --- Level 1/2 book info ---
    "bid_1_price",
    "ask_1_price",
    "bid_1_volume",
    "ask_1_volume",
    "bid_1_volume_delta",
    "ask_1_volume_delta",

    "bid_2_price",
    "bid_2_volume",
    "ask_2_price",
    "ask_2_volume",

    # --- Book imbalance ---
    "depth_imb",

    # --- TXF features ---
    "txf_price_delta_1s",
    "txf_price_delta_5s",
    "txf_volume_delta_1s",
    "txf_volume_delta_5s",
]

for d, g in df.groupby("trade_date", sort=True):
    g = g.copy()
    g.loc[:, price_features] = g[price_features].ffill()
    # 確保 bucket_start 為 datetime 型態
    if not np.issubdtype(g["bucket_start"].dtype, np.datetime64):
        g["bucket_start"] = pd.to_datetime(g["bucket_start"])

    label_raw = g["relative_vwap_rise_fall"].iloc[-1]
    if label_raw not in (-1, 0, 1):
        continue
    label = int(label_raw + 1)  # -1/0/1 -> 0/1/2

    bucket_time = g["bucket_start"].dt.time
    mask = (bucket_time >= start_t) & (bucket_time <= end_t)
    cut = g[mask]
    if cut.empty:
        continue

    seq_len = int(SEQ_LEN_OVERRIDE) if SEQ_LEN_OVERRIDE else len(cut)
    if len(cut) < seq_len:
        continue

    window = cut.iloc[-seq_len:][price_features].to_numpy()
    sequences.append(window)
    targets.append(label)
    dates.append(d)

sequences = np.array(sequences)
targets = np.array(targets)

# 切 train/test，避免 test 為空
split_idx = max(int(len(sequences) * 0.8), 1)
train_Xp, test_Xp = sequences[:split_idx], sequences[split_idx:]
train_yp, test_yp = targets[:split_idx], targets[split_idx:]
if len(test_Xp) == 0:
    test_Xp, test_yp = train_Xp, train_yp

model, metrics, pred_df = train_price_lstm_cls3(
    train_Xp, train_yp, test_Xp, test_yp,
    use_val = USE_VAL,
    early_stop = EARLY_STOP,
    patience = PATIENCE,
    min_delta = MIN_DELTA,
    batch_size = BATCH,
    epochs = EPOCHS,
    lr = LR,
    hidden = HIDDEN,
    layers = LAYERS,
    dropout = DROPOUT,
    seed = RANDOM_SEED,
    scale = SCALE,
)

display(pd.DataFrame(metrics["report"]).T)
display(pred_df)
[Cls3] Epoch 01 | train = 1.0526 | val = 0.9762
[Cls3] Epoch 05 | train = 0.7167 | val = 0.6531
[Cls3] Epoch 10 | train = 0.6355 | val = 0.7556
[Cls3] Epoch 15 | train = 0.5298 | val = 0.8538
[Cls3] Epoch 20 | train = 0.3973 | val = 0.9376
[Cls3] Epoch 25 | train = 0.3053 | val = 1.1378
[Cls3] Epoch 30 | train = 0.1890 | val = 1.7168
Pred class distribution: {np.int64(2): np.int64(41)}
precision	recall	f1-score	support
0	0.000000	0.000000	0.000000	27.000000
1	0.000000	0.000000	0.000000	1.000000
2	0.317073	1.000000	0.481481	13.000000
accuracy	0.317073	0.317073	0.317073	0.317073
macro avg	0.105691	0.333333	0.160494	41.000000
weighted avg	0.100535	0.317073	0.152665	41.000000
actual_cls	pred_cls	prob_down	prob_flat	prob_up
0	0	2	0.381346	0.004601	0.614053
1	2	2	0.379972	0.004855	0.615173
2	0	2	0.325502	0.005092	0.669406
3	2	2	0.404022	0.004734	0.591244
4	2	2	0.384539	0.004762	0.610699
5	0	2	0.400575	0.005084	0.594341
6	0	2	0.401880	0.005161	0.592959
7	0	2	0.380370	0.004720	0.614911
8	0	2	0.393984	0.004720	0.601296
9	2	2	0.343314	0.004806	0.651881
10	2	2	0.370629	0.004950	0.624421
11	2	2	0.378671	0.005330	0.615999
12	0	2	0.360620	0.005328	0.634052
13	0	2	0.364769	0.004759	0.630472
14	0	2	0.404748	0.004854	0.590397
15	0	2	0.380980	0.005403	0.613618
16	2	2	0.412721	0.004926	0.582354
17	0	2	0.402878	0.004661	0.592461
18	0	2	0.377142	0.005079	0.617779
19	2	2	0.383467	0.005570	0.610963
20	0	2	0.387654	0.004991	0.607355
21	2	2	0.368987	0.005299	0.625714
22	2	2	0.337997	0.004864	0.657139
23	0	2	0.342358	0.004945	0.652697
24	0	2	0.381947	0.005104	0.612949
25	0	2	0.351049	0.016348	0.632602
26	0	2	0.388682	0.005121	0.606197
27	0	2	0.346550	0.005325	0.648125
28	0	2	0.316312	0.005016	0.678672
29	0	2	0.358246	0.005349	0.636406
30	1	2	0.397783	0.005442	0.596775
31	0	2	0.391670	0.005118	0.603212
32	0	2	0.413743	0.008887	0.577369
33	2	2	0.380504	0.009240	0.610256
34	0	2	0.405043	0.008900	0.586058
35	2	2	0.411490	0.008636	0.579874
36	2	2	0.404867	0.008682	0.586451
37	0	2	0.411402	0.009074	0.579524
38	0	2	0.402238	0.009161	0.588601
39	0	2	0.398587	0.008777	0.592636
40	0	2	0.392850	0.008835	0.598315
 
block
# LSTM 訓練（每日一樣本，13:20~13:24，將 4 分鐘的 bucket 分成多個 block，對每 block 取平均形成序列）
SAMPLE_START_TIME = "13:20:00"
BUCKET_MS = 50
BLOCKS_PER_DAY = 20                # 想要的 block 數（4 分鐘 / BLOCKS_PER_DAY）
BUCKETS_PER_BLOCK = int((4 * 60 * 1000 / BUCKET_MS) // BLOCKS_PER_DAY)  # 4 分鐘總 bucket / block 數

BATCH = 32
EPOCHS = 30
LR = 1e-3
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.2
PATIENCE = 5
MIN_DELTA = 1e-4
USE_VAL = True
EARLY_STOP = False
RANDOM_SEED = 42
SCALE = True

start_dt = datetime.strptime(SAMPLE_START_TIME, "%H:%M:%S")
start_t = start_dt.time()
end_exclusive = (start_dt + timedelta(minutes=4)).time()  # 13:24:00，使用 < 排除結束點

sequences, targets, dates = [], [], []
price_features = [
    "last_price", "price_delta", "return_1s", "return_5s",
    "time_to_close_s",
    "volume", "vol_1s", "vol_5s", "volume_1s", "volume_5s",
    "io_imb_1s", "io_imb_5s",
    "bid_1_price", "ask_1_price",
    "bid_1_volume", "ask_1_volume",
    "bid_1_volume_delta", "ask_1_volume_delta",
    "bid_2_price", "bid_2_volume", "ask_2_price", "ask_2_volume",
    "depth_imb",
]

f = df
if not hasattr(f["bucket_start"], 'dt'):
    f["bucket_start"] = pd.to_datetime(f["bucket_start"])
f = f.sort_values(["trade_date", "bucket_start"]).reset_index(drop=True)

for d, g in df.groupby("trade_date", sort=True):
    g = g.copy()
    g.loc[:, price_features] = g[price_features].ffill()
    label_raw = g["relative_vwap_rise_fall"].iloc[-1]
    if label_raw not in (-1, 0, 1):
        continue
    label = int(label_raw + 1)

    bt = g["bucket_start"].dt.time
    cut = g[(bt >= start_t) & (bt < end_exclusive)]
    if len(cut) != BUCKETS_PER_BLOCK * BLOCKS_PER_DAY:
        continue

    blocks = []
    for i in range(BLOCKS_PER_DAY):
        start_idx = i * BUCKETS_PER_BLOCK
        end_idx = start_idx + BUCKETS_PER_BLOCK
        block = cut.iloc[start_idx:end_idx][price_features]
        block_mean = block.mean()
        blocks.append(block_mean.to_numpy())

    seq = np.vstack(blocks)  # shape (BLOCKS_PER_DAY, feature_size)
    sequences.append(seq)
    targets.append(label)
    dates.append(d)

sequences = np.array(sequences)  # shape: (num_days, BLOCKS_PER_DAY, feature_size)
targets = np.array(targets)

if len(sequences) == 0:
    raise ValueError("No samples collected: 時間窗內筆數不足或分段不滿 BLOCKS_PER_DAY，請調整條件。")

# 依交易日切 train/test，同一天不拆分
unique_dates = sorted(set(dates))
split_d = max(int(len(unique_dates) * 0.8), 1)
train_dates = set(unique_dates[:split_d])
test_dates = set(unique_dates[split_d:])

train_idx = [i for i, d in enumerate(dates) if d in train_dates]
test_idx  = [i for i, d in enumerate(dates) if d in test_dates]

train_X = sequences[train_idx]
test_X  = sequences[test_idx]
train_y = targets[train_idx]
test_y  = targets[test_idx]
if len(test_X) == 0:
    test_X, test_y = train_X, train_y

model, metrics, pred_df = train_price_lstm_cls3(
    train_X, train_y, test_X, test_y,
    use_val = USE_VAL,
    early_stop = EARLY_STOP,
    patience = PATIENCE,
    min_delta = MIN_DELTA,
    batch_size = BATCH,
    epochs = EPOCHS,
    lr = LR,
    hidden = HIDDEN,
    layers = LAYERS,
    dropout = DROPOUT,
    seed = RANDOM_SEED,
    scale = SCALE,
)

display(pd.DataFrame(metrics["report"]).T)
display(pred_df)
[Cls3] Epoch 01 | train = 1.0535 | val = 0.9936
[Cls3] Epoch 05 | train = 0.7381 | val = 0.7258
[Cls3] Epoch 10 | train = 0.6437 | val = 0.7155
[Cls3] Epoch 15 | train = 0.5620 | val = 0.8026
[Cls3] Epoch 20 | train = 0.4398 | val = 0.7569
[Cls3] Epoch 25 | train = 0.3451 | val = 1.2191
[Cls3] Epoch 30 | train = 0.2898 | val = 1.5934
Pred class distribution: {np.int64(2): np.int64(41)}
precision	recall	f1-score	support
0	0.000000	0.000000	0.000000	27.000000
1	0.000000	0.000000	0.000000	1.000000
2	0.317073	1.000000	0.481481	13.000000
accuracy	0.317073	0.317073	0.317073	0.317073
macro avg	0.105691	0.333333	0.160494	41.000000
weighted avg	0.100535	0.317073	0.152665	41.000000
actual_cls	pred_cls	prob_down	prob_flat	prob_up
0	0	2	0.343310	0.002480	0.654210
1	2	2	0.337413	0.001355	0.661232
2	0	2	0.322986	0.001902	0.675111
3	2	2	0.391851	0.002288	0.605861
4	2	2	0.383142	0.002045	0.614813
5	0	2	0.378504	0.001664	0.619832
6	0	2	0.442096	0.001703	0.556201
7	0	2	0.431959	0.001708	0.566333
8	0	2	0.329113	0.002854	0.668032
9	2	2	0.391568	0.002172	0.606261
10	2	2	0.342831	0.002210	0.654959
11	2	2	0.366338	0.002203	0.631459
12	0	2	0.348692	0.007585	0.643723
13	0	2	0.349912	0.002084	0.648004
14	0	2	0.386149	0.001677	0.612174
15	0	2	0.362946	0.002204	0.634849
16	2	2	0.350424	0.002999	0.646576
17	0	2	0.373745	0.003755	0.622499
18	0	2	0.391205	0.002229	0.606566
19	2	2	0.374674	0.002331	0.622995
20	0	2	0.412067	0.002538	0.585395
21	2	2	0.381178	0.002953	0.615869
22	2	2	0.399374	0.002304	0.598322
23	0	2	0.367910	0.002774	0.629316
24	0	2	0.385053	0.003196	0.611750
25	0	2	0.462118	0.001910	0.535972
26	0	2	0.390964	0.003182	0.605854
27	0	2	0.378683	0.004282	0.617035
28	0	2	0.383344	0.003179	0.613477
29	0	2	0.398780	0.003947	0.597273
30	1	2	0.400531	0.009304	0.590164
31	0	2	0.398796	0.004145	0.597059
32	0	2	0.408964	0.004183	0.586853
33	2	2	0.396745	0.004230	0.599024
34	0	2	0.401134	0.004306	0.594560
35	2	2	0.403899	0.004719	0.591383
36	2	2	0.403198	0.003710	0.593092
37	0	2	0.419193	0.003204	0.577604
38	0	2	0.399834	0.004190	0.595976
39	0	2	0.385262	0.005224	0.609515
40	0	2	0.401247	0.003830	0.594923
 
other
label
print("all", df.groupby("trade_date")["relative_rise_fall"].last().value_counts())
display("train", dict(zip(*np.unique(train_yp, return_counts = True))))
display("test", dict(zip(*np.unique(test_yp, return_counts = True))))
'all'
relative_rise_fall
 1    75
-1    67
 0    59
Name: count, dtype: int64
'train'
{np.int64(0): np.int64(47),
 np.int64(1): np.int64(47),
 np.int64(2): np.int64(65)}
'test'
{np.int64(0): np.int64(19),
 np.int64(1): np.int64(12),
 np.int64(2): np.int64(9)}
 
鴻海
read
pg_uri = "postgresql+psycopg2://devuser:DevPass123!@localhost:5432/spot"
pg_engine = create_engine(pg_uri, future = True)

start_time = "12:30:00"
end_time = "13:30:00"
start_date_override: str | None = "2025-01-01"
end_date_override: str | None = "2025-11-30"

if end_time:
    slice_query = text("""
    SELECT *
    FROM public.t2317_features
    WHERE transaction_time::time BETWEEN :start_time AND :end_time
      AND trade_date BETWEEN COALESCE(:start_date, trade_date) AND COALESCE(:end_date, trade_date)
    ORDER BY trade_date, transaction_time
    """)
else:
    slice_query = text("""
    SELECT *
    FROM public.t2317_features
    WHERE transaction_time::time >= :start_time
      AND trade_date BETWEEN COALESCE(:start_date, trade_date) AND COALESCE(:end_date, trade_date)
    ORDER BY trade_date, transaction_time
    """)

params = {
    "start_time": start_time,
    "start_date": start_date_override,
    "end_date": end_date_override,
}
if end_time:
    params["end_time"] = end_time

with pg_engine.connect() as conn:
    features = pd.read_sql_query(slice_query, conn, params = params)

features = features.sort_values(["trade_date", "transaction_time"]).reset_index(drop = True)
block
SEQ_LEN_PRICE = 1000
BATCH_PRICE = 64
EPOCHS_PRICE = 30
LR_PRICE = 1e-3
HIDDEN_PRICE = 128
LAYERS_PRICE = 2
DROPOUT_PRICE = 0.2
PATIENCE_PRICE = 5
MIN_DELTA_PRICE = 1e-4
USE_VAL_PRICE = True
EARLY_STOP_PRICE = False
RANDOM_SEED = 42
SCALE = True
sequences, targets, dates = [], [], []

price_features = [
    "in_out", "sum_io_rolling_10", "price", "mid_price", "mid_price_delta", "spread",
    "spread_delta", "price_delta", "momentum", "volatility_10",
    "hl_range", "imbalance", "delta_bid_1_volume", "delta_ask_1_volume",
    "vol_diff", "vol_ma_10", "time_to_close", "time_since_prev_trade",
    "trade_intensity", "imb_spread", "ret_vol", "vol_imb", "is_trade",
]

for d, g in features.groupby("trade_date", sort = True):
    g = g.copy()
    g.loc[:, price_features] = g[price_features].ffill()

    label_raw = g["relative_rise_fall"].iloc[-1]
    if label_raw not in (-1, 0, 1):
        continue
    label = int(label_raw + 1)  # -1/0/1 -> 0/1/2

    cutoff = pd.Timestamp(d) + pd.Timedelta(hours=13, minutes=24)
    cut = g[g["transaction_time"] <= cutoff]
    if len(cut) < SEQ_LEN_PRICE:
        continue

    window = cut.iloc[-SEQ_LEN_PRICE:][price_features].to_numpy()
    sequences.append(window)
    targets.append(label)
    dates.append(d)

sequences = np.array(sequences)
targets = np.array(targets)

block_size = 50

def block_pool(x, block):
    usable = (x.shape[1] // block) * block  # 只取可整除的部分
    x_cut = x[:, :usable, :]
    x_view = x_cut.reshape(x.shape[0], -1, block, x.shape[2])
    return x_view.mean(axis = 2)

# 切 train/test，並在套用 block pooling 後仍沿用 train_Xp/test_Xp 命名，方便後續流程和 SHAP
split_idx = max(int(len(sequences) * 0.8), 1)
train_Xp_raw, test_Xp_raw = sequences[:split_idx], sequences[split_idx:]
train_yp, test_yp = targets[:split_idx], targets[split_idx:]

train_Xp = block_pool(train_Xp_raw, block_size)
test_Xp  = block_pool(test_Xp_raw,  block_size)

model, metrics, pred_df = train_price_lstm_cls3(
    train_Xp, train_yp, test_Xp, test_yp,
    use_val = USE_VAL_PRICE,
    early_stop = EARLY_STOP_PRICE,
    patience = PATIENCE_PRICE,
    min_delta = MIN_DELTA_PRICE,
    batch_size = BATCH_PRICE,
    epochs = EPOCHS_PRICE,
    lr = LR_PRICE,
    hidden = HIDDEN_PRICE,
    layers = LAYERS_PRICE,
    dropout = DROPOUT_PRICE,
    seed = RANDOM_SEED,
    scale = SCALE
)

display(pd.DataFrame(metrics["report"]).T)
display(pred_df)
[Cls3] Epoch 01 | train = 1.1125 | val = 1.1079
[Cls3] Epoch 05 | train = 1.0233 | val = 1.2213
[Cls3] Epoch 10 | train = 0.8887 | val = 1.3596
[Cls3] Epoch 15 | train = 0.7364 | val = 1.7232
[Cls3] Epoch 20 | train = 0.6192 | val = 2.1792
[Cls3] Epoch 25 | train = 0.4824 | val = 2.4975
[Cls3] Epoch 30 | train = 0.3649 | val = 2.8595
Pred class distribution: {np.int64(0): np.int64(41)}
precision	recall	f1-score	support
0	0.341463	1.000000	0.509091	14.000000
1	0.000000	0.000000	0.000000	14.000000
2	0.000000	0.000000	0.000000	13.000000
accuracy	0.341463	0.341463	0.341463	0.341463
macro avg	0.113821	0.333333	0.169697	41.000000
weighted avg	0.116597	0.341463	0.173836	41.000000
actual_cls	pred_cls	prob_down	prob_flat	prob_up
0	0	0	0.355347	0.333382	0.311271
1	2	0	0.359468	0.326204	0.314327
2	2	0	0.358456	0.327912	0.313633
3	1	0	0.359076	0.326813	0.314111
4	2	0	0.356275	0.329935	0.313790
5	2	0	0.356430	0.331075	0.312495
6	0	0	0.357594	0.329189	0.313217
7	2	0	0.358966	0.327545	0.313488
8	1	0	0.356478	0.331183	0.312340
9	1	0	0.358933	0.327314	0.313754
10	2	0	0.355661	0.331323	0.313015
11	0	0	0.357287	0.330051	0.312662
12	2	0	0.358869	0.326265	0.314865
13	2	0	0.356476	0.327647	0.315877
14	0	0	0.356760	0.330446	0.312794
15	1	0	0.357642	0.327712	0.314646
16	2	0	0.358436	0.327705	0.313858
17	1	0	0.361078	0.323055	0.315867
18	0	0	0.360531	0.323741	0.315728
19	2	0	0.356212	0.329323	0.314465
20	1	0	0.358600	0.325982	0.315418
21	1	0	0.358275	0.326710	0.315015
22	1	0	0.357162	0.329921	0.312917
23	2	0	0.356916	0.327687	0.315397
24	1	0	0.356530	0.329448	0.314023
25	2	0	0.353911	0.331774	0.314315
26	1	0	0.357951	0.327423	0.314627
27	1	0	0.366005	0.321548	0.312447
28	2	0	0.354977	0.332866	0.312157
29	0	0	0.356914	0.329449	0.313638
30	0	0	0.352763	0.333874	0.313363
31	0	0	0.355978	0.329645	0.314377
32	1	0	0.356083	0.329333	0.314584
33	0	0	0.358587	0.325421	0.315992
34	0	0	0.359822	0.324335	0.315842
35	0	0	0.356985	0.328178	0.314837
36	0	0	0.358387	0.325705	0.315908
37	1	0	0.354590	0.332837	0.312573
38	0	0	0.356350	0.329233	0.314417
39	0	0	0.357698	0.326288	0.316014
40	1	0	0.356078	0.327764	0.316158
n_features_price = len(price_features)
if SCALE:
    block_scaler = StandardScaler()
    train_Xp_scaled = block_scaler.fit_transform(train_Xp.reshape(-1, n_features_price)).reshape(train_Xp.shape)
    test_Xp_scaled = block_scaler.transform(test_Xp.reshape(-1, n_features_price)).reshape(test_Xp.shape)
else:
    train_Xp_scaled = train_Xp.copy()
    test_Xp_scaled = test_Xp.copy()

block_train_tensor = torch.tensor(train_Xp_scaled, dtype = torch.float32)
block_test_tensor = torch.tensor(test_Xp_scaled, dtype = torch.float32)
block_device = next(model.parameters()).device

block_shap_importance = compute_lstm_shap(
    model = model,
    train_tensor = block_train_tensor,
    test_tensor = block_test_tensor,
    device = block_device,
    feature_names = price_features,
    max_background = 20,
    max_samples = 10
)
display(block_shap_importance.head(10).to_frame(name = 'mean|SHAP|'))
mean|SHAP|
price	0.000116
trade_intensity	0.000100
volatility_10	0.000099
mid_price	0.000086
imbalance	0.000086
vol_imb	0.000085
momentum	0.000080
in_out	0.000066
is_trade	0.000062
imb_spread	0.000059
觀察
read & format
start_date = "2025-01-01"
end_date = "2025-11-14"
start_time = "13:20:00"
end_time = "13:24:00"

# spot
df = GetPostgreData(
    sources = [{
        "label": "t2330_features",
        "uri": "postgresql+psycopg2://devuser:DevPass123!@localhost:5432/spot",
        "table": "t2330_features",
        "date_col": "trade_date",
        "time_col": "transaction_time",
        "rename_map": {
            "price": "price_2330"
        },
        "order_by": ("trade_date", "transaction_time"),
    }],
    start_date = start_date,
    end_date = end_date,
    start_time = start_time,
    end_time = end_time
)
t2330_features = df["t2330_features"]

# # future
# df = GetPostgreData(
#     sources = [{
#         "label": "tTX_df",
#         "uri": "postgresql+psycopg2://devuser:DevPass123!@localhost:5432/future",
#         "table": '"tTXF"',
#         "rename_map": {
#             "dd": "trade_date",
#             "tt": "transaction_time",
#             "f01": "stock_id",
#             "f04": "price"
#         },
#         "order_by": ("dd", "tt")
#     }],
#     start_date = start_date,
#     end_date = end_date,
#     start_time = start_time,
#     end_time = end_time
# )
# tTX_df = df["tTX_df"]
# tTX_df = filter_near_contract(
#     df = tTX_df,
#     settle_csv = "../../note/data/settle_date_TX.csv",
#     sid_col = "stock_id",
#     trade_date_col = "trade_date",
#     settle_type = "month"
# )

# df = GetPostgreData(
#     sources = [{
#         "label": "cdf_df",
#         "uri": "postgresql+psycopg2://devuser:DevPass123!@localhost:5432/future",
#         "table": '"CDF"',
#         "rename_map": {
#             "dd": "trade_date",
#             "tt": "transaction_time",
#             "f01": "stock_id",
#             "f04": "price"
#         },
#         "order_by": ("dd", "tt")
#     }],
#     start_date = start_date,
#     end_date = end_date,
#     start_time = start_time,
#     end_time = end_time
# )
# cdf_df = df["cdf_df"]
# cdf_df = filter_near_contract(
#     df = cdf_df,
#     settle_csv = "../../note/data/settle_date_TX.csv",
#     sid_col = "stock_id",
#     trade_date_col = "trade_date",
#     settle_type = "month"
# )
t2330_features['trade_date'] = pd.to_datetime(t2330_features['trade_date'])
t2330_features['transaction_time'] = pd.to_datetime(t2330_features['transaction_time'])
tTX_df['trade_date'] = pd.to_datetime(tTX_df['trade_date'])
tTX_df['transaction_time'] = pd.to_datetime(tTX_df['transaction_time'])
cdf_df['trade_date'] = pd.to_datetime(cdf_df['trade_date'])
cdf_df['transaction_time'] = pd.to_datetime(cdf_df['transaction_time'])

key_cols = ['trade_date', 'transaction_time']

def _add_suffix_except(df, suffix):
    rename_map = {c: f"{c}{suffix}" for c in df.columns if c not in key_cols}
    return df.rename(columns = rename_map)

tTX_df_suf = _add_suffix_except(tTX_df, '_tx')
cdf_df_suf = _add_suffix_except(cdf_df, '_cdf')

all_df = (
    t2330_features
    .merge(tTX_df_suf, on = key_cols, how = 'outer')
    .merge(cdf_df_suf, on = key_cols, how = 'outer')
    .sort_values(key_cols)
    .reset_index(drop = True)
)

price_cols = [c for c in all_df.columns if 'price' in c.lower()]
dfs_by_date = {}
for d, g in all_df.groupby('trade_date', sort = True):
    g_copy = g.copy()
    if price_cols:
        g_copy.loc[:, price_cols] = g_copy[price_cols].ffill()
    dfs_by_date[d] = g_copy
relative rise fall
加權指數高低
twii = pd.read_pickle("../../note/data/twii.pkl")
twii["20_ma"] = twii["close"].rolling(20).mean()
twii["60_ma"] = twii["close"].rolling(60).mean()
twii["120_ma"] = twii["close"].rolling(120).mean()

twii["hl_20"] = (twii["close"] / twii["20_ma"]) - 1
twii["hl_60"] = (twii["close"] / twii["60_ma"]) - 1
twii["hl_120"] = (twii["close"] / twii["20_ma"]) - 1

df = t2330_features.copy()
df["trade_date"] = pd.to_datetime(df["trade_date"])
twii["date"] = pd.to_datetime(twii["date"])
twii.rename(columns = {"date": "trade_date"}, inplace = True)

label_df = (
    df[["trade_date", "relative_rise_fall"]]
    .drop_duplicates("trade_date")
)

twii = twii.merge(
    label_df,
    on = "trade_date",
    how = "left"
)
twii[["hl_20", "hl_60", "hl_120"]] = twii[["hl_20", "hl_60", "hl_120"]].shift(1)

def plot_and_bin(col, label_col="relative_rise_fall", q=10):
    tmp = twii[[col, label_col]].replace([np.inf, -np.inf], np.nan)
    tmp = tmp.dropna(subset=[col, label_col])
    tmp[label_col] = tmp[label_col].astype(float)

    tmp = tmp.sort_values(col)
    plot(tmp, ly=label_col, x=col, ry=col)

    tmp["bin"] = pd.qcut(tmp[col], q=q, duplicates="drop")
    res = (
        tmp.groupby("bin", observed=False)[label_col]
           .value_counts(normalize=True)
           .unstack(fill_value=0)
           .T
    )
    res.columns = [f"bin_{i}" for i in range(res.shape[1])]
    display(res)

plot_and_bin("hl_20")
plot_and_bin("hl_60")
plot_and_bin("hl_120")
bin_0	bin_1	bin_2	bin_3	bin_4	bin_5	bin_6	bin_7	bin_8	bin_9
relative_rise_fall										
-1.0	0.388889	0.388889	0.411765	0.333333	0.166667	0.294118	0.277778	0.176471	0.555556	0.388889
0.0	0.222222	0.222222	0.294118	0.277778	0.222222	0.470588	0.444444	0.294118	0.166667	0.277778
1.0	0.388889	0.388889	0.294118	0.388889	0.611111	0.235294	0.277778	0.529412	0.277778	0.333333
bin_0	bin_1	bin_2	bin_3	bin_4	bin_5	bin_6	bin_7	bin_8	bin_9
relative_rise_fall										
-1.0	0.214286	0.642857	0.230769	0.214286	0.285714	0.153846	0.357143	0.230769	0.428571	0.500000
0.0	0.142857	0.000000	0.384615	0.357143	0.357143	0.307692	0.428571	0.307692	0.214286	0.357143
1.0	0.642857	0.357143	0.384615	0.428571	0.357143	0.538462	0.214286	0.461538	0.357143	0.142857
bin_0	bin_1	bin_2	bin_3	bin_4	bin_5	bin_6	bin_7	bin_8	bin_9
relative_rise_fall										
-1.0	0.388889	0.388889	0.411765	0.333333	0.166667	0.294118	0.277778	0.176471	0.555556	0.388889
0.0	0.222222	0.222222	0.294118	0.277778	0.222222	0.470588	0.444444	0.294118	0.166667	0.277778
1.0	0.388889	0.388889	0.294118	0.388889	0.611111	0.235294	0.277778	0.529412	0.277778	0.333333
收盤前強度
df = t2330_features.copy()

# --- 取 13:20~13:24 這段時間（使用 bucket_start） ---
mask = df["transaction_time"].dt.time.between(
    pd.to_datetime("13:20:00").time(),
    pd.to_datetime("13:24:00").time()
)
cut = df[mask].copy()

# --- 確保 in_out 為 numeric ---
cut["in_out"] = pd.to_numeric(cut["in_out"], errors="coerce").fillna(0)
cut["abs_in_out"] = cut["in_out"].abs()

# --- 計算每日 bucket 數（不能寫死，要依實際資料） ---
bucket_count = cut.loc[cut["in_out"] != 0].groupby("trade_date").size().rename("n_bucket")

# --- 每日聚合 ---
agg = cut.groupby("trade_date").agg(
    sum_in_out=("in_out", "sum"),
    sum_abs=("abs_in_out", "sum"),
    relative_rise_fall=("relative_rise_fall", "last")
).reset_index()

# 合併 bucket 數
agg = agg.merge(bucket_count, on="trade_date", how="left")

# --- DUI 公式（方向 × 密度）---
agg["dui"] = np.where(
    agg["sum_abs"] == 0,
    0,
    (agg["sum_in_out"] / agg["sum_abs"]) * (agg["sum_abs"] / agg["n_bucket"])
)

agg.sort_values("dui", inplace = True)
agg = agg.iloc[1: ]
plot(agg, "relative_rise_fall", ry = "dui")

# 依 DUI 分 10 個 quantile bin
agg['dui_bin'] = pd.qcut(agg['dui'], q=10, labels=False, duplicates='drop')

# 各 bin 的樣本數與漲/平/跌比例
bin_summary = agg.groupby('dui_bin').agg(
    count=('dui', 'size'),
    avg_dui=('dui', 'mean'),
    rise_ratio=('relative_rise_fall', lambda x: (x == 1).mean()),
    flat_ratio=('relative_rise_fall', lambda x: (x == 0).mean()),
    fall_ratio=('relative_rise_fall', lambda x: (x == -1).mean()),
).reset_index()

# 橫向呈現：每列一個指標，每欄為 bin
horizontal = bin_summary.set_index('dui_bin').T
display(horizontal)
dui_bin	0	1	2	3	4	5	6	7	8	9
count	20.000000	20.000000	20.000000	19.000000	20.000000	20.000000	19.000000	20.00000	20.000000	20.000000
avg_dui	-3.311283	-1.546648	-0.968943	-0.472255	-0.090391	0.181869	0.443378	0.84305	1.352303	2.904431
rise_ratio	0.350000	0.450000	0.350000	0.210526	0.500000	0.350000	0.526316	0.25000	0.450000	0.350000
flat_ratio	0.400000	0.200000	0.250000	0.421053	0.250000	0.250000	0.263158	0.30000	0.150000	0.400000
fall_ratio	0.250000	0.350000	0.400000	0.368421	0.250000	0.400000	0.210526	0.45000	0.400000	0.250000
relative vwap rise fall
加權指數高低
# 對 hl_20/hl_60/hl_120 做單變量排序 + qcut 分組 (清理 NaN/Inf)
import numpy as np

if 'twii' not in globals():
    raise RuntimeError('twii 不存在，請先準備好 twii DataFrame')

def plot_and_bin(col, label_col='relative_vwap_rise_fall', q=10):
    tmp = twii[['trade_date', col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
    tmp = tmp.sort_values(col).reset_index(drop=True)
    plot(tmp, ly=label_col, x=col, ry=col)  # 使用既有 plot 函數
    tmp['bin'] = pd.qcut(tmp[col], q=q, duplicates='drop')
    res = tmp.groupby('bin', observed=False)[label_col] \
                        .value_counts(normalize=True) \
                        .unstack(fill_value=0)
    res = res.T  # 行=label, 列=bin
    res.columns = [f'bin{i}' for i in range(len(res.columns))]
    display(res.round(3))

plot_and_bin('hl_20')
plot_and_bin('hl_60')
plot_and_bin('hl_120')
bin	0	1	2	3	4	5	6	7	8	9
label_-1.0	0.474	0.444	0.556	0.444	0.278	0.556	0.500	0.333	0.667	0.526
label_0.0	0.105	0.056	0.000	0.000	0.000	0.056	0.056	0.056	0.000	0.053
label_1.0	0.421	0.500	0.444	0.556	0.722	0.389	0.444	0.611	0.333	0.421
bin	0	1	2	3	4	5	6	7	8	9
label_-1.0	0.267	0.571	0.571	0.286	0.5	0.143	0.714	0.357	0.571	0.733
label_0.0	0.133	0.071	0.000	0.000	0.0	0.071	0.071	0.000	0.071	0.000
label_1.0	0.600	0.357	0.429	0.714	0.5	0.786	0.214	0.643	0.357	0.267
bin	0	1	2	3	4	5	6	7	8	9
label_-1.0	0.0	0.250	0.556	0.375	0.667	0.500	0.875	0.444	0.625	0.778
label_0.0	0.0	0.125	0.000	0.000	0.000	0.125	0.000	0.111	0.000	0.000
label_1.0	1.0	0.625	0.444	0.625	0.333	0.375	0.125	0.444	0.375	0.222
收盤前買賣力道
df = t2330_features.copy()

# --- 取 13:20~13:24 這段時間（使用 bucket_start） ---
mask = df["bucket_start"].dt.time.between(
    pd.to_datetime("13:20:00").time(),
    pd.to_datetime("13:24:00").time()
)
cut = df[mask].copy()

# --- 確保 in_out 為 numeric ---
cut["sum_in_out"] = pd.to_numeric(cut["sum_in_out"], errors="coerce").fillna(0)
cut["abs_in_out"] = cut["sum_in_out"].abs()

# --- 計算每日 bucket 數（不能寫死，要依實際資料） ---
bucket_count = cut.loc[cut["sum_in_out"] != 0].groupby("trade_date").size().rename("n_bucket")

# --- 每日聚合 ---
agg = cut.groupby("trade_date").agg(
    sum_in_out=("sum_in_out", "sum"),
    sum_abs=("abs_in_out", "sum"),
    relative_vwap_rise_fall=("relative_vwap_rise_fall", "last")
).reset_index()

# 合併 bucket 數
agg = agg.merge(bucket_count, on="trade_date", how="left")

# --- DUI 公式（方向 × 密度）---
agg["dui"] = np.where(
    agg["sum_abs"] == 0,
    0,
    (agg["sum_in_out"] / agg["sum_abs"]) * (agg["sum_abs"] / agg["n_bucket"])
)

agg.sort_values("dui", inplace = True)
agg = agg.iloc[1: ]
plot(agg, "relative_vwap_rise_fall", ry = "dui")

# 依 DUI 分 10 個 quantile bin
agg['dui_bin'] = pd.qcut(agg['dui'], q=10, labels=False, duplicates='drop')

# 各 bin 的樣本數與漲/平/跌比例
bin_summary = agg.groupby('dui_bin').agg(
    count=('dui', 'size'),
    avg_dui=('dui', 'mean'),
    rise_ratio=('relative_vwap_rise_fall', lambda x: (x == 1).mean()),
    flat_ratio=('relative_vwap_rise_fall', lambda x: (x == 0).mean()),
    fall_ratio=('relative_vwap_rise_fall', lambda x: (x == -1).mean()),
).reset_index()

# 橫向呈現：每列一個指標，每欄為 bin
horizontal = bin_summary.set_index('dui_bin').T
display(horizontal)
dui_bin	0	1	2	3	4	5	6	7	8	9
count	21.000000	21.000000	20.000000	21.000000	23.000000	18.000000	21.000000	20.000000	21.000000	21.000000
avg_dui	-3.512184	-1.616368	-1.010061	-0.436007	-0.055575	0.147420	0.441376	0.883653	1.393926	3.134107
rise_ratio	0.380952	0.523810	0.400000	0.380952	0.347826	0.388889	0.619048	0.500000	0.523810	0.666667
flat_ratio	0.000000	0.000000	0.050000	0.000000	0.391304	0.000000	0.000000	0.000000	0.047619	0.000000
fall_ratio	0.619048	0.476190	0.550000	0.619048	0.260870	0.611111	0.380952	0.500000	0.428571	0.333333
看收盤前價格變化
start_date_plot = pd.Timestamp("2025-09-15")
end_date_plot   = pd.Timestamp("2025-09-30")

use_random = False
sample_n = 5
np.random.seed(42)

# 時間範圍（可視需求調整）
start_t = pd.to_datetime("13:00:00").time()
end_t   = pd.to_datetime("13:30:00").time()

# 篩選落在區間內的日期
eligible_dates = [d for d in dfs_by_date.keys() if start_date_plot <= d <= end_date_plot]
if use_random and eligible_dates:
    sample_dates = list(np.random.choice(eligible_dates, size=min(sample_n, len(eligible_dates)), replace=False))
else:
    sample_dates = eligible_dates

if not sample_dates:
    print("指定區間內沒有可用日期。")
else:
    for d in sample_dates:
        g = dfs_by_date[d].copy()
        g["price"] = g.get("price_2330", g.get("price", pd.Series())).replace(0, np.nan).ffill()
        mask = (g["transaction_time"].dt.time >= start_t) & (g["transaction_time"].dt.time <= end_t)
        g_slice = g.loc[mask]
        if g_slice.empty:
            continue
        plot(
            g_slice,
            ["price", "price_cdf_cdf"],
            x = "transaction_time",
            ry = "price_tx_tx",
            ry_dashed = False,
            sub_ly = ["bid_1_volume", "ask_1_volume"],
        )
看台積電跳動影響台指期
tTX_df = pd.read_csv("../data/tTX.csv")
tTX_df.rename(columns = {
    "dd": "trade_date",
    "tt": "transaction_time",
    "f04": "price_tx",
    "f05": "trade_volume",
    "f11": "cum_trade_volume"
}, inplace = True)
tTX_df["transaction_time"] = pd.to_datetime(
    tTX_df["trade_date"].astype(str) + " " + tTX_df["transaction_time"].astype(str)
)
tTX_df["stock_id"] = "TX"

pg_uri = "postgresql+psycopg2://devuser:DevPass123!@localhost:5432/t2330"
pg_engine = create_engine(pg_uri, future = True)

start_time = "13:00:00"
end_time = None  # 若只需 13:30 前資料，可設定為 "13:30:00"
start_date_override: str | None = "2025-09-19"
end_date_override: str | None = "2025-10-15"

if end_time:
    slice_query = text("""
    SELECT *
    FROM public.t2330_features
    WHERE transaction_time::time BETWEEN :start_time AND :end_time
      AND trade_date BETWEEN COALESCE(:start_date, trade_date) AND COALESCE(:end_date, trade_date)
    ORDER BY trade_date, transaction_time
    """)
else:
    slice_query = text("""
    SELECT *
    FROM public.t2330_features
    WHERE transaction_time::time >= :start_time
      AND trade_date BETWEEN COALESCE(:start_date, trade_date) AND COALESCE(:end_date, trade_date)
    ORDER BY trade_date, transaction_time
    """)

params = {
    "start_time": start_time,
    "start_date": start_date_override,
    "end_date": end_date_override,
}
if end_time:
    params["end_time"] = end_time

with pg_engine.connect() as conn:
    raw_slice = pd.read_sql_query(slice_query, conn, params = params)

postgre_df = raw_slice.sort_values(["trade_date", "transaction_time"]).reset_index(drop = True)
def tick_size(price: float) -> float:
    if price < 10: return 0.01
    if price < 50: return 0.05
    if price < 100: return 0.1
    if price < 500: return 0.5
    if price < 1000: return 1
    return 5

# 股票、期貨資料
tsmc = all_df[["transaction_time", "price"]].dropna().sort_values("transaction_time")
tsmc = tsmc[tsmc["price"] != 0]  # drop zero prices to avoid false jumps
tx   = all_df[["transaction_time", "price_tx"]].dropna().sort_values("transaction_time")

# 計算 tick 數
tsmc = tsmc[tsmc["price"] != 0]
tsmc["price_diff"] = tsmc["price"].diff()
tsmc["tick_used"]  = tsmc["price"].shift().apply(tick_size)

tsmc["tick_count_raw"] = tsmc["price_diff"] / tsmc["tick_used"]
tsmc["tick_count"] = tsmc["tick_count_raw"].round().astype("Int64")

# # 合理 tick 跳動
# good_jump = np.isclose(
#     tsmc["price_diff"].abs(),
#     (tsmc["tick_used"] * tsmc["tick_count"].abs()).astype(float)
# )

# events = tsmc.loc[ good_jump & (tsmc["tick_count"].abs() >= 1) ].copy()
# events["direction"]  = np.sign(events["tick_count"]).astype(int)
# events["abs_ticks"]  = events["tick_count"].abs().astype(int)
# events["event_ts"]   = events["transaction_time"]

# # 整理 TX
# tx_sorted = tx.rename(columns={"transaction_time": "tx_time", "price_tx": "tx_price"}).sort_values("tx_time")

# # === 重點 1：算多個 horizon（訊號反應在 10–200 ms）===
# horizons = [50, 100, 150, 200, 250, 500]  # 毫秒

# summary_list = {}

# for H in horizons:
#     h_delta = pd.to_timedelta(H, unit="ms")
#     e = events.copy()
#     e["ts_after"] = e["event_ts"] + h_delta

#     # === 重點 2：正確的對齊方式（不能用 nearest） ===
#     # 事件當下 → backward（不能偷看未來）
#     e = pd.merge_asof(
#         e.sort_values("event_ts"),
#         tx_sorted,
#         left_on="event_ts",
#         right_on="tx_time",
#         direction="backward",
#         tolerance=pd.Timedelta("1s")
#     ).rename(columns={"tx_price": "tx_before"})

#     # horizon 之後 → backward（取得 horizon 之前最近成交）
#     e = pd.merge_asof(
#         e.sort_values("ts_after"),
#         tx_sorted,
#         left_on="ts_after",
#         right_on="tx_time",
#         direction="backward",
#         tolerance=pd.Timedelta("1s")
#     ).rename(columns={"tx_price": "tx_after"})

#     e = e.dropna(subset=["tx_before", "tx_after"])
#     e["tx_move"] = e["tx_after"] - e["tx_before"]
#     e["tx_move_per_tick"] = e["tx_move"] / e["tick_count"]

#     summary = e.groupby("direction")["tx_move_per_tick"].agg(["count", "mean", "median"])
#     summary_list[H] = summary

# # 輸出所有 horizon
# for H, s in summary_list.items():
#     print(f"\n=== Horizon {H} ms ===")
#     print(s)
print("rows tsmc:", len(tsmc))
print("nonzero diffs:", (tsmc["price_diff"] != 0).sum())

events_raw = tsmc.loc[
    np.isclose(tsmc["price_diff"].abs(),
               (tsmc["tick_used"] * (tsmc["price_diff"]/tsmc["tick_used"]).round().abs()))
    & (tsmc["price_diff"].abs() >= tsmc["tick_used"])
]
print("events_raw:", len(events_raw))

print("events after merge_asof:", len(events))
print("missing tx_before:", events["tx_before"].isna().sum(),
      "missing tx_after:", events["tx_after"].isna().sum())
rows tsmc: 94334
nonzero diffs: 26234
events_raw: 26233
events after merge_asof: 7607
missing tx_before: 0 missing tx_after: 0
# 將序列按 block_size 壓縮（每 block_size 筆做均值），適用於 train_Xp/test_Xp
block_size = 50
if train_Xp.shape[1] % block_size != 0 or test_Xp.shape[1] % block_size != 0:
    print(f"警告: 序列長度不能被 block_size 整除, 將截斷到最接近的整數倍")

def block_pool(x, block):
    usable = (x.shape[1] // block) * block
    x_cut = x[:, :usable, :]
    x_view = x_cut.reshape(x.shape[0], -1, block, x.shape[2])
    return x_view.mean(axis=2)

train_Xp_block = block_pool(train_Xp, block_size)
test_Xp_block = block_pool(test_Xp, block_size)
print("原序列 shape:", train_Xp.shape, "-> 壓縮後:", train_Xp_block.shape)
print("test shape:", test_Xp.shape, "->", test_Xp_block.shape)
other
 