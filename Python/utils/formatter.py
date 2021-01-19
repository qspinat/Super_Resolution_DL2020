from datetime import datetime
from common.constants import CHECKPOINTS_FOLDER


def format_checkpoint_name():
    now = datetime.now()
    checkpoint_file = CHECKPOINTS_FOLDER + f'checkpoints_{now.hour}{now.minute}{now.second}_{now.day}{now.month}{now.year}_best.pth'
    return checkpoint_file
