from dms_efficientnet import EffDmsAlgorithm
import unittest
from timm.models.efficientnet import efficientnet_b0


class TestEffDms(unittest.TestCase):
    def test_init(self):
        model = efficientnet_b0(drop_path_rate=0.3, drop_rate=0.2)
        print(model)

        algo = EffDmsAlgorithm(
            model,
            mutator_kwargs=dict(
                dtp_mutator_cfg=dict(
                    parse_cfg=dict(
                        demo_input=dict(
                            input_shape=(1, 3, 240, 240),
                        ),
                    )
                ),
            ),
        )
        print(algo.mutator.info())
