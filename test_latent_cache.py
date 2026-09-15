"""CPU tests of actual cache and collate functions extracted from trainer AST."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import torch

SOURCE = Path(__file__).resolve().parent/'phase3/train_cogvideox_lora.py'


class Distribution:
    def __init__(self):
        self.calls = 0

    def sample(self):
        self.calls += 1
        return torch.randn(1, 2, 3, 4, 4)


class CacheTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.name in {'cache_video_distributions', 'collate_fn'}]
        scope = dict(torch=torch, vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=1.0)))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)
        self.cache = scope['cache_video_distributions']
        self.collate = scope['collate_fn']

    def dataset(self, mode='center'):
        return SimpleNamespace(instance_video_paths=[Path('/fixture/a'), Path('/fixture/./a'), Path('/fixture/b')],
            instance_video_indices=[[0, 1], [0, 1], [0, 1]], video_reshape_mode=mode,
            _decode_video=Mock(side_effect=lambda i:i))

    def test_shared_distribution_independent_sampling(self):
        data = self.dataset(); progress = Mock(); encode = Mock(side_effect=lambda _:Distribution())
        self.assertEqual(self.cache(data, encode, progress), 2)
        self.assertEqual(data._decode_video.call_count, 2)
        self.assertEqual(progress.update.call_count, 3)
        self.assertEqual(len(data.instance_latents), 3)
        self.assertIs(data.instance_latents[0], data.instance_latents[1])
        self.assertIsNot(data.instance_latents[0], data.instance_latents[2])
        torch.manual_seed(21)
        batch = self.collate([dict(instance_video=d, instance_prompt=str(i)) for i,d in enumerate(data.instance_latents)])
        self.assertEqual(batch['prompts'], ['0', '1', '2'])
        self.assertEqual(tuple(batch['videos'].shape), (3, 3, 2, 4, 4))
        self.assertFalse(torch.equal(batch['videos'][0], batch['videos'][1]))
        self.assertEqual(data.instance_latents[0].calls, 2)

    def test_different_frame_indices_not_shared(self):
        data=self.dataset();data.instance_video_indices[1]=[2,3]
        self.assertEqual(self.cache(data, lambda _:Distribution(), Mock()), 3)

    def test_randomized_crop_not_shared(self):
        for mode in ('random', 'none'):
            data=self.dataset(mode)
            self.assertEqual(self.cache(data, lambda _:Distribution(), Mock()), 3)
            self.assertIsNot(data.instance_latents[0], data.instance_latents[1])

    def test_full_weighted_lists(self):
        root = SOURCE.parent.parent/'p0_audit_results/paired_video_full_lossless_v1'
        for arm, expected_rows in [('A0',500),('A1',1319),('A2',500),('A3',1319)]:
            paths=[root/arm/p for p in (root/arm/'videos.txt').read_text().splitlines()]
            data=SimpleNamespace(instance_video_paths=paths,instance_video_indices=[list(range(49)) for _ in paths],
                video_reshape_mode='center',_decode_video=Mock(side_effect=lambda i:i))
            encode=Mock(side_effect=lambda _:Distribution())
            self.assertEqual(self.cache(data,encode,Mock()),500)
            self.assertEqual(encode.call_count,500)
            self.assertEqual(len(data.instance_latents),expected_rows)


if __name__=='__main__':unittest.main()
