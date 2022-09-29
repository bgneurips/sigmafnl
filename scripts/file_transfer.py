import rich
import fabric
import logging
import pathlib
from tqdm import tqdm
from patchwork.transfers import rsync
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.DEBUG)
handler = RichHandler()
logger.addHandler(handler)

class SCP:

    def __init__(self, host_destination='symmetry', username='ugiri'):

        self.client = fabric.Connection(host=host_destination, user=username, forward_agent=True)


    def start_transfer(self, source_list, destination_list):
        logger.debug('Starting transfer ...')        
        for (local, remote) in tqdm(zip(source_list, destination_list)):
            logger.debug(f'Transfering {local.split("/")[-1]} to {remote}')
            self.client.put(local=local, remote=remote)

class Rsync:

    def __init__(self, host_destination='symmetry', username='ugiri'):

        self.client = fabric.Connection(host=host_destination, user=username, forward_agent=True)

    def start_transfer(self, source_list, destination):
        destination = pathlib.Path(destination)
        assert destination.is_dir()

        logger.debug('Starting rsync transfer ...')        
        
        for local in tqdm(source_list):
            local = pathlib.Path(local)
            assert local.is_dir()
            logger.debug(f'Transfering {local.split("/")[-1]} to {remote}')
            rsync(self.client, source=local, target=destination)


