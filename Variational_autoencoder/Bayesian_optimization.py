import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import GPyOpt
from torchvision.utils import make_grid
from Model import VAE
import torch


class FacialComposit:
    def __init__(self, seg_decoder, latent_size):
        self.latent_size = latent_size
        self.decode = seg_decoder
        self.samples = None
        self.images = None
        self.rating = None

    def _get_image(self, latent):

        img = self.decode(torch.tensor(latent).type(torch.float32))
        img = make_grid(img).permute(1, 2, 0).detach().numpy()

        return img

    @staticmethod
    def _show_images(images, titles):
        assert len(images) == len(titles)
        clear_output()
        plt.figure(figsize=(3 * len(images), 3))
        n = len(titles)
        for i in range(n):
            plt.subplot(1, n, i + 1)

            plt.imshow(images[i])
            plt.title(str(titles[i]))
            plt.axis('off')
        plt.show()

    @staticmethod
    def _draw_border(image, w=2):
        bordred_image = image.copy()
        bordred_image[:, :w] = [1, 0, 0]
        bordred_image[:, -w:] = [1, 0, 0]
        bordred_image[:w, :] = [1, 0, 0]
        bordred_image[-w:, :] = [1, 0, 0]
        return bordred_image

    def query_initial(self, n_start=10, select_top=6):
        '''
        Creates initial points for Bayesian optimization
        Generate *n_start* random images and asks user to rank them.
        Gives maximum score to the best image and minimum to the worst.
        :param n_start: number of images to rank initialy.
        :param select_top: number of images to keep
        '''
        self.samples = np.zeros([select_top, self.latent_size])  ### YOUR CODE HERE (size: select_top x 64 x 64 x 3)
        self.images = np.zeros([select_top, 128, 128, 3])  ### YOUR CODE HERE (size: select_top x 64 x 64 x 3)
        self.rating = np.zeros([select_top])  ### YOUR CODE HERE (size: select_top)

        ### YOUR CODE:
        ### Show user some samples (hint: use self._get_image and input())
        images = []
        titles = []
        scores = []
        codes = []  # list of latent

        for i in range(n_start):
            latent_code = np.random.normal(size=[self.latent_size])  # create latent sample
            codes.append(latent_code)  # add the latent code to the list of latent codes

            images.append(self._get_image(latent_code))
            titles.append("Zdjęcie " + str(i + 1))

        self._show_images(images, titles)

        print('Initial images/points for bayesian optimization')
        user_scores = input('Please rate between 1 (worst) to 10 (best), seprate values by -: ')
        for sc in user_scores.split('-'):
            scores.append(int(sc))

        # verification for user mistake
        assert (len(scores) == len(images))

        # Sort in descending order and get indices
        indices_desc = sorted(range(len(scores)), key=lambda z: -scores[z])

        # keep the first select_top images
        for i, idx in enumerate(indices_desc[:select_top]):
            self.samples[i] = codes[idx]
            self.images[i] = images[idx]
            self.rating[i] = scores[idx]

        # Check that tensor sizes are correct
        np.testing.assert_equal(self.rating.shape, [select_top])
        np.testing.assert_equal(self.images.shape, [select_top, 128, 128, 3])
        np.testing.assert_equal(self.samples.shape, [select_top, self.latent_size])

    def evaluate(self, candidate):

        '''
        Queries candidate vs known image set.
        Adds candidate into images pool.
        :param candidate: latent vector of size 1xlatent_size
        '''
        initial_size = len(self.images)

        ### YOUR CODE HERE
        ## Show user an image and ask to assign score to it.
        ## You may want to show some images to user along with their scores
        ## You should also save candidate, corresponding image and rating
        image = self._get_image(candidate[0])

        images = list(self.images[:3])  #
        images.append(image)

        titles = list(self.rating[:3])  #
        titles.append("candidate")

        self._show_images(images, titles)

        candidate_rating = int(input("Please rate the candidate image:"))

        self.images = np.vstack((self.images, np.array([image])))
        self.rating = np.hstack((self.rating, np.array([candidate_rating])))
        self.samples = np.vstack((self.samples, candidate))

        assert len(self.images) == initial_size + 1
        assert len(self.rating) == initial_size + 1
        assert len(self.samples) == initial_size + 1
        return candidate_rating

    def optimize(self, n_iter=6, w=4, acquisition_type='MPI', acquisition_par=0.3):
        if self.samples is None:
            self.query_initial()

        bounds = [{'name': 'z_{0:03d}'.format(i),
                   'type': 'continuous',
                   'domain': (-w, w)}
                  for i in range(self.latent_size)]

        optimizer = GPyOpt.methods.BayesianOptimization(f=self.evaluate, domain=bounds,
                                                        acquisition_type=acquisition_type,
                                                        acquisition_par=acquisition_par,
                                                        exact_eval=False,
                                                        model_type='GP',
                                                        X=self.samples,
                                                        Y=self.rating[:, None],
                                                        maximize=True)

        optimizer.run_optimization(max_iter=n_iter, eps=-1, evaluations_file="Save.txt",
                                   report_file="report.txt", models_file="model.txt")

    def get_best(self):
        index_best = np.argmax(self.rating)
        return self.images[index_best]

    def draw_best(self, title=''):
        index_best = np.argmax(self.rating)
        image = self.images[index_best]
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    model = VAE().load_from_checkpoint(
        checkpoint_path=r"Path\file.ckpt")

    decoder = model.decode

    composit = FacialComposit(decoder, 64)
    composit.optimize()
    composit.draw_best('Best')
