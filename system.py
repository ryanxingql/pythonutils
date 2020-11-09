import os
import time

def get_timestr():
    """Return current time str."""
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())