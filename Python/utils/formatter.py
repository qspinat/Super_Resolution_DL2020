from datetime import datetime
from common.constants import CHECKPOINTS_FOLDER


def format_best_checkpoint_name():
    now = datetime.now()
    checkpoint_file = CHECKPOINTS_FOLDER + f'checkpoints_{now.strftime("%d%m%Y")}_{now.strftime("%H%M%S")}_best.pth'
    return checkpoint_file

def format_current_checkpoint_name():
    now = datetime.now()
    checkpoint_file = CHECKPOINTS_FOLDER + f'checkpoints_{now.strftime("%d%m%Y")}_{now.strftime("%H%M%S")}_current.pth'
    return checkpoint_file
