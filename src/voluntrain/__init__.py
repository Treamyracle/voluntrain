# Expose the main classes directly
from .host import ElasticHost as Host
from .worker import ElasticWorker as Worker

__all__ = ["Host", "Worker"]