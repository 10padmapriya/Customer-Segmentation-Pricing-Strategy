from pathlib import Path
import urllib.request
import zipfile
import pandas as pd

def download_raw_data(out_path: str = "../data/raw_retail.csv") -> pd.DataFrame:
    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True)

    zip_path = out_path.parent / "online_retail_ii.zip"
    xlsx_path = out_path.parent / "online_retail_II.xlsx"

    if not xlsx_path.exists():
        print("Downloading UCI Online Retail II...")
        url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            print("Files in zip:", z.namelist())
            z.extractall(out_path.parent)
        print("Download complete!")

    print("Reading Excel file...")
    df1 = pd.read_excel(xlsx_path, sheet_name="Year 2009-2010")
    df2 = pd.read_excel(xlsx_path, sheet_name="Year 2010-2011")
    df = pd.concat([df1, df2], ignore_index=True)

    print(f"  Shape     : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns   : {list(df.columns)}")

    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return df
