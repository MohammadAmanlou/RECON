"""Executable phase-2 tests: python -m unittest knight_a.test_assessment -v."""
import unittest
import numpy as np
from knight_a import assessment as a


class AssessmentTests(unittest.TestCase):
    def test_priors_and_posterior(self):
        for family in ('pairwise', 'hyperedge'):
            p = a.prior(family, a.DEFAULT)
            self.assertAlmostEqual(p.sum(), 1.)
            self.assertTrue((p > 0).all())
            like = a.likelihood(a.DEFAULT)
            for correct in (True, False):
                self.assertAlmostEqual(a.posterior_update(p, like[:, 0], correct).sum(), 1.)
        unary = {3: (1,)}
        np.testing.assert_allclose(a.prior('pairwise', a.DEFAULT, unary), a.prior('hyperedge', a.DEFAULT, unary))

    def test_budget_replay_and_no_repeat(self):
        world = a.DevWorld('matched', 100, 3)
        for method in a.METHODS:
            def run():
                return a.run_learner(method, a.DEFAULT, lambda j: world.answers[0, j], a.rng_for(100, 5))
            p, h = run()
            p2, h2 = run()
            np.testing.assert_array_equal(p, p2)
            self.assertEqual(h, h2)
            self.assertEqual(len(h), 8)
            self.assertEqual(len({x['family'] for x in h}), 8)
            self.assertEqual(len({x['id'] for x in h}), 8)
            self.assertTrue(np.isfinite(p).all())

    def test_world_replay_and_split(self):
        w1, w2 = [a.DevWorld('mismatch', 300, 4) for _ in range(2)]
        np.testing.assert_array_equal(w1.mastery, w2.mastery)
        np.testing.assert_array_equal(w1.answers, w2.answers)
        for regime, seeds in a.PROTOCOL['worlds']['held_out'].items():
            with self.assertRaises(ValueError):
                a.DevWorld(regime, seeds[0], 1)

    def test_policy_isolation(self):
        # Change hidden truth, graph and all future potential answers; same observed prefix.
        w1, w2 = [a.DevWorld('matched', 100, 1) for _ in range(2)]
        for method in a.METHODS:
            w2.mastery[:] = 1-w1.mastery
            w2.true_parents = {}
            w2.answers[:] = ~w1.answers
            _, h1 = a.run_learner(method, a.DEFAULT, lambda j: w1.answers[0,j], a.rng_for(100, 5), 2)
            first = next(i for i,q in enumerate(a.BANK) if q['id'] == h1[0]['id'])
            w2.answers[0,first] = w1.answers[0,first]
            _, h2 = a.run_learner(method, a.DEFAULT, lambda j: w2.answers[0,j], a.rng_for(100, 5), 2)
            self.assertEqual(h1[0], h2[0])
            self.assertEqual(h1[1]['id'], h2[1]['id'])

    def test_discriminative_toy(self):
        # First skill uncertain, second known. IG must prefer the informative item.
        states = np.array([[0,1],[1,1]], dtype=float)
        q = np.eye(2)
        p = np.array([.5,.5])
        like = a.likelihood(a.DEFAULT, q, states)
        choice = a.select('hyperedge_ig', p, like, np.ones(2, dtype=bool), a.rng_for(1,5), q, states)
        self.assertEqual(choice, 0)
        updated = a.posterior_update(p, like[:,choice], True)
        self.assertGreater((updated@states)[0], .5)


if __name__ == '__main__':
    unittest.main()
