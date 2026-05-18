from .base import Task
from .alfworld import AlfWorldTask
from .webshop import WebShopTask
try:
    from .intercode_sql import IntercodeSQLTask
except ModuleNotFoundError:
    IntercodeSQLTask = None
