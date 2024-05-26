from pyhive import presto
from config import settings

from typing import List

cursor = presto.connect(host='porta.data-engineering.myteksi.net', 
                        username=f'{settings.presto_username};cloud=aws&mode=adhoc',
                        password=settings.presto_password,
                        catalog='hive',
                        schema='public',
                        port=443,
                        protocol='https').cursor()

def list_schemas() -> List[str]:
    cursor.execute("show schemas")
    return cursor.fetchall()

def list_tables_for_schema(schema: str) -> List[str]:
    cursor.execute(f"show tables from {schema}")
    return cursor.fetchall()

def run_query(query: str) -> str:
    cursor.execute(query)
    return cursor.fetchall()