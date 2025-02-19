import logging
import select
import time

logger = logging.getLogger("imdclient.IMDClient")

# NOTE: think of other edge cases as well- should be robust
def parse_host_port(filename):
    if not filename.startswith("imd://"):
        raise ValueError("IMDReader: URL must be in the format 'imd://host:port'")
    
    # Check if the format is correct
    parts = filename.split("imd://")[1].split(":")
    if len(parts) == 2:
        host = parts[0] 
        try:
            port = int(parts[1])
            return (host, port)
        except ValueError:
            raise ValueError("IMDReader: Port must be an integer")
    else:
        raise ValueError("IMDReader: URL must be in the format 'imd://host:port'")
