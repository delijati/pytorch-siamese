import sys

import argparse
import unittest
import torch
import numpy as np

from roux_siamese_few_shot.contrastive import ContrastiveLoss
from torch.autograd import Variable, gradcheck

torch.set_default_tensor_type('torch.DoubleTensor')


def run_tests():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, default=123)
    args, remaining = parser.parse_known_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    remaining = [sys.argv[0]] + remaining
    unittest.main(argv=remaining)


class TestContrastive(unittest.TestCase):
    def setUp(self):
        self.x0 = torch.from_numpy(
            # np.array(
            #     [[0.39834601, 0.6656751], [-0.44211167, -0.95197892],
            #      [0.52718359, 0.69099563], [-0.36314946, -0.07625845],
            #      [-0.53021497, -0.67317766]],
            #     dtype=np.float32)
            np.random.uniform(-1, 1, (5, 2)).astype(np.float32)
        )
        self.x1 = torch.from_numpy(
            # np.array(
            #     [[0.73587674, 0.98970324], [-0.9245277, 0.93210953],
            #      [-0.32989913, 0.36705822], [0.25636896, 0.10106555],
            #      [-0.11412049, 0.80171216]],
            #     dtype=np.float32)
            np.random.uniform(-1, 1, (5, 2)).astype(np.float32)
        )
        self.t = torch.from_numpy(
            # np.array(
            #     [1, 0, 1, 1, 0], dtype=np.float32)
            np.random.randint(0, 2, (5,)).astype(np.float32)
        )
        self.margin = 1

    def test_contrastive_loss(self):
        input1 = Variable(torch.randn(4, 4), requires_grad=True)
        input2 = Variable(torch.randn(4, 4), requires_grad=True)
        target = Variable(torch.randn(4), requires_grad=True)
        tml = ContrastiveLoss(margin=self.margin)
        self.assertTrue(
            gradcheck(lambda x1, x2, t: tml.forward(x1, x2, t), (
                input1, input2, target)))

    def test_contrastive_loss_value(self):
        x0_val = Variable(self.x0)
        x1_val = Variable(self.x1)
        t_val = Variable(self.t)
        tml = ContrastiveLoss(margin=self.margin)
        loss = tml.forward(x0_val, x1_val, t_val)
        self.assertIsNotNone(loss.data.item())
        self.assertEqual(loss.data.numpy().dtype, np.float32)
        loss_value = float(loss.data.numpy())

        # Compute expected value
        loss_expect = 0
        for i in range(self.x0.size()[0]):
            x0d, x1d, td = self.x0[i], self.x1[i], self.t[i]
            d = torch.sum(torch.pow(x0d - x1d, 2))
            if td == 1:  # similar pair
                loss_expect += d
            elif td == 0:  # dissimilar pair
                loss_expect += max(1 - np.sqrt(d), 0)**2
        loss_expect /= 2.0 * self.t.size()[0]
        print("expected %s got %s" % (loss_expect, loss_value))
        self.assertAlmostEqual(loss_expect, loss_value, places=5)


if __name__ == '__main__':
    run_tests()
