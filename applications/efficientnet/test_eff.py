from dms_efficientnet import EffDmsAlgorithm, TeacherStudentDistillNetDIST
import unittest
from timm.models.efficientnet import efficientnet_b0
import timm
import torch


class TestEffDms(unittest.TestCase):
    def test_init(self):
        model = efficientnet_b0(drop_path_rate=0.3, drop_rate=0.2)
        algo = EffDmsAlgorithm(
            model,
            mutator_kwargs=dict(
                dtp_mutator_cfg=dict(
                    parse_cfg=dict(
                        demo_input=dict(
                            input_shape=(1, 3, 224, 224),
                        ),
                    )
                ),
            ),
        )
        print(algo.mutator.info())

    def test_init_fold(self):
        model = efficientnet_b0(drop_path_rate=0.3, drop_rate=0.2)
        algo = EffDmsAlgorithm.init_fold_algo(
            model,
            mutator_kwargs=dict(
                dtp_mutator_cfg=dict(
                    parse_cfg=dict(
                        demo_input=dict(
                            input_shape=(1, 3, 224, 224),
                        ),
                    )
                ),
            ),
        )
        print(algo.mutator.info())

    def test_distill(self):
        teacher = timm.create_model("efficientnet_b4", pretrained=True)
        model = efficientnet_b0(drop_path_rate=0.3, drop_rate=0.2)
        a = torch.rand(2, 3, 224, 224)

        distill_model = TeacherStudentDistillNetDIST(
            teacher, model, teacher_input_image_size=288, input_image_size=224
        )
        distill_model(a)
        distill_model.compute_ts_distill_loss()
