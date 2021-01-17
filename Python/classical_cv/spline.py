import time
from tqdm import tqdm
import numpy as np
from scipy.ndimage import zoom


class SplineEvaluation():
    def __init__(self, order=0, display_time=False):
        assert(order >= 0 and order <= 5)
        self.order = order
        self.display_time = display_time

    def interpolate(self, img):
        start = time.time()
        interpolate_result = zoom(img, zoom=2, order=self.order)
        if len(img.shape) == 3:
            interpolate_result = interpolate_result[:,:,[0,2,4]]
        
        if self.display_time:
            print(f'Took {time.time() - start}s to compute a spline interpolation of order {self.order} on an image of shape {img.shape}')
        
        return interpolate_result

    def evaluate_dataset(self, dataset):
        total_loss = 0
        for data in tqdm(dataset):
            image = data['image'].permute(1,2,0).numpy()
            label = data['label'].permute(1,2,0).numpy()

            interpolated = self.interpolate(image)
            loss = np.linalg.norm(label - interpolated)
            total_loss += loss

        print(f'Mean L2 loss on the dataset : {total_loss / len(dataset)}')
