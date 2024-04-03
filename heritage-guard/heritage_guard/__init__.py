from dagster import Definitions, load_assets_from_modules
from dagster_duckdb_pandas import DuckDBPandasIOManager
from os import path
from . import assets
from .CONSTANTS import ROOT_PATH

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    resources={
        "duck_io_manager": DuckDBPandasIOManager(
            database=path.join(ROOT_PATH, 'database.duckdb')
        )
    },
)
