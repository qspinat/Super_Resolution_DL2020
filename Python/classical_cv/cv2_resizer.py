import time
from tqdm import tqdm
import numpy as np
import cv2


class Cv2ResizerEvaluation():
    def __init__(self, display_time=False):
        self.display_time = display_time

    def interpolate(self, img, output_size):
        start = time.time()
        interpolate_result = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
        
        if self.display_time:
            print(f'Took {time.time() - start}s to compute a linear interpolation on an image of shape {img.shape}')
        
        return interpolate_result

    def evaluate_dataset(self, dataset):
        total_loss = 0
        for data in tqdm(dataset):
            image = data['image'].permute(1,2,0).numpy()
            label = data['label'].permute(1,2,0).numpy()

            interpolated = self.interpolate(image, output_size=label.shape[:2])
            loss = np.linalg.norm(label - interpolated)
            total_loss += loss

        print(f'Mean L2 loss on the dataset : {total_loss / len(dataset)}')
