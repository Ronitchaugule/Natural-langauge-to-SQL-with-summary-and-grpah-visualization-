import pandas as pd
import sqlite3

def load_file_to_sqlite(uploaded_file, table_name="data_table"):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError("Only CSV and .xlsx files are supported.")

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql(table_name, conn, index=False, if_exists="replace")

    return conn, table_name, df

def get_schema(connection, table_name):
    cursor = connection.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    cols = cursor.fetchall()

    schema_lines = []
    for col in cols:
        column_name = col[1]
        column_type = col[2]
        schema_lines.append(f'"{column_name}" {column_type}')

    schema_text = f"Table {table_name} columns:\n"
    schema_text += "\n".join(schema_lines)

    return schema_text
