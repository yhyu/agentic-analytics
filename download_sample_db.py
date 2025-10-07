import os
import sqlite3
import urllib.request
import pandas as pd
from pathlib import Path


def download_sales_data(download_path: str = 'db') -> None:
    Path(download_path).mkdir(parents=True, exist_ok=True)
    down_url = 'https://data.montgomerycountymd.gov/api/views/v76h-r7br/rows.csv?accessType=DOWNLOAD'
    download_file = os.path.join(download_path, 'Warehouse_and_Retail_Sales.csv')
    urllib.request.urlretrieve(down_url, download_file)

def csv2sqlite(csv_file: str, sqlite_file: str):
    schema = """CREATE TABLE "warehouse_and_retail_sales" ("year" int, "month" int, "supplier" text, "item_code" text, "item_description" text, "item_type" text, "retail_sales" real, "retail_transfers" real, "warehouse_sales" real, primary key("year", "month", "supplier", "item_code"))"""
    con = sqlite3.connect(sqlite_file)
    cur = con.cursor()
    cur.execute(schema)

    df = pd.read_csv(csv_file)
    cur.executemany("""INSERT INTO warehouse_and_retail_sales
    (year, month, supplier, item_code, item_description, item_type, retail_sales, retail_transfers, warehouse_sales)
    VALUES (?,?,?,?,?,?,?,?,?)""", df.to_dict(orient='split')['data'])
    con.commit()
    con.close()
    os.remove(csv_file)

if __name__ == "__main__":
    # download sales data
    print('download warehouse_and_retail_sales data...(it may take a few seconds.)')
    download_sales_data()

    # convert to sqlite
    print('convert csv file to sqlite db...')
    csv2sqlite('db/Warehouse_and_Retail_Sales.csv', 'db/warehouse_and_retail_sales.sqlite')

    print('prepare example db done!')
