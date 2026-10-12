import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('warm_jar',Path(__file__).resolve().parents[1]/'scripts/remote/warm-jar-policy.py')
warm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warm)


class JarWarmupTests(unittest.TestCase):
    def test_warmup_seed_cannot_equal_a_maniguard_episode_seed(self):
        # benchmark.py masks crc32(seed:scene) with 0x7fffffff.
        self.assertGreater(warm.WARMUP_SEED,0x7fffffff)
        self.assertLessEqual(warm.WARMUP_SEED,0xffffffff)

    def test_wrong_policy_or_checkpoint_is_rejected(self):
        expected = {'serve_config':'pi05-base_datagen_v1_jar_joint_2cam_lora',
                    'checkpoint':'/workspace/checkpoints/pi05-jar/7400'}
        warm.validate_metadata(expected)
        for key in expected:
            wrong = dict(expected,**{key:'different'})
            with self.assertRaises(ValueError):warm.validate_metadata(wrong)


if __name__=='__main__':unittest.main()
