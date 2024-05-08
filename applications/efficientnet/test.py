import torch
from torch import Tensor
from timm.models.efficientnet import (
    efficientnet_b0,
    EfficientNet,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
)
import time
import torch.nn as nn
from mmrazor.models.utils.expandable_utils import make_channel_divisible
import ptflops
import json
from dms_eff import DMSMutator
import random
from mmrazor.implementations.pruning.dms.core.mutable import (
    DMSMutableMixIn,
    MutableBlocks,
)
import copy
import re
from collections import OrderedDict
from timm.layers.norm_act import BatchNormAct2d
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d

import unittest
from collections import defaultdict
import json
import torch.nn.functional as F
from dms_eff import (
    EffDmsAlgorithm,
    Predictor,
    replaece_InvertedResidual_forward,
    replaece_Mult_forward,
    AddModule,
    MultModule,
    isBasicModule,
)
import thop


def rm_bn_from_net(net):
    for m in net.modules():
        if isinstance(m, BatchNormAct2d):

            def forward_wrapper(m):
                def forward(x):
                    return m.act(x)

                return forward

            m.forward = forward_wrapper(m)

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x


@torch.no_grad()
def measure_net_latency(
    net, input_shape=(3, 224, 224), clean=True, device=torch.device("cuda:0")
):
    net = net.eval().to(device)
    # remove bn from graph
    rm_bn_from_net(net)
    batch_size = 8
    n_warmup = 100
    n_sample = 200
    images = [
        torch.zeros([batch_size] + list(input_shape), device=device),
    ]
    measured_latency = {"warmup": [], "sample": []}
    net.eval()
    if isinstance(net, AddModule):
        images.append(images[0])
    if isinstance(net, MultModule):
        images.append(images[0].sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True))

    with torch.no_grad():
        for i in range(n_warmup):
            net(*images)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        torch.cuda.synchronize()
        starter.record()
        for i in range(n_sample):
            net(*images)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)

    return curr_time / n_sample, measured_latency


def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = OrderedDict()
            remove_all_forward_hooks(child)


def collect_info(model):
    # register
    def forward_hook(m, i, o):
        try:
            m.in_shape = i[0].shape
            m.out_shape = o.shape
        except:
            pass

    for n, m in model.named_modules():
        m.register_forward_hook(forward_hook)

    x = torch.rand([2, 3, 224, 224]).cuda()
    model(x)
    remove_all_forward_hooks(model)


# replace add mult


def make_divisor(n):
    import math

    for d in [64, 32, 16, 8, 4, 2]:
        if d < n // 10:
            if n % d == 0:
                return n // d
    return n


class TestLatency(unittest.TestCase):
    def test_split(self):
        replaece_InvertedResidual_forward()
        replaece_Mult_forward()

        # model: EfficientNet = efficientnet_b0().cuda().eval()
        model: EfficientNet = efficientnet_b4().cuda().eval()
        print(measure_net_latency(model)[0])

        predictor = Predictor()
        predictor.build_net()

        # collec info
        collect_info(model)

        # meaure

        quick_measure = lambda m: measure_net_latency(m, input_shape=m.in_shape[1:])[0]

        modules = [
            *[b for state in model.blocks for b in state],
            # *model.blocks,
            model.conv_head,
            model.global_pool,
            model.conv_stem,
            model.classifier,
        ]

        all_latency = []
        all_sub_latency = []
        pred_ls = []

        for module in modules:
            module_latency = quick_measure(module)
            all_latency.append(module_latency)

            # test sub modules

            sub_ls = []
            for n, m in module.named_modules():
                if isBasicModule(m) and hasattr(m, "in_shape"):
                    sub_ls.append(quick_measure(m))
                    sub = sub_ls[-1]
                    lx = predictor.predict_m(m).item()
                    pred_ls.append(lx)
                    if abs(lx - sub) / sub > 0.2:
                        print(predictor.parse_m(m))
                        print(n)
                        print(m)
                        print(lx, sub)
                        print()

            sub_l = sum(sub_ls)
            all_sub_latency.append(sub_l)
            # if abs(sub_l - module_latency) / module_latency > 0.2:
            #     print("block error", sum(sub_ls), module_latency)
            #     print(module)
            #     print()

        print(sum(all_latency))
        print(sum(all_sub_latency))
        print(sum(pred_ls))
        # print(model)

    def test_collect_modules(self):
        replaece_InvertedResidual_forward()
        replaece_Mult_forward()

        model: EfficientNet = efficientnet_b4().cuda().eval()
        collect_info(model)

        basic_modules = defaultdict(
            lambda: {
                "max_in": 0,
                "max_out": 0,
                "min_in": int(1e4),
                "min_out": int(1e4),
            }
        )

        for m in model.modules():
            if isBasicModule(m) and hasattr(m, "in_shape"):
                args, inc, outc = Predictor.parse_m(m)
                config = basic_modules[args]
                config["max_in"] = max(inc, config["max_in"])
                config["max_out"] = max(outc, config["max_out"])
                config["min_in"] = min(inc, config["min_in"])
                config["min_out"] = min(outc, config["min_out"])

        print(json.dumps(basic_modules, indent=4))
        print(len(basic_modules))
        for args in basic_modules:
            config = basic_modules[args]
            config["records"] = []
            if args.startswith("conv2d"):
                inc, outc = config["max_in"], config["max_out"]
                k = int(re.findall(r"k(\d+)", args)[0])
                s = int(re.findall(r"s(\d+)", args)[0])
                p = int(re.findall(r"p(\d+)", args)[0])
                r = int(re.findall(r"r(\d+)", args)[0])
                dw = int(re.findall(r"dw(\d+)", args)[0])
                bias = int(re.findall(r"b(\d+)", args)[0])
                if dw == 1:
                    assert inc == outc
                    n = make_divisor(inc)
                    for i in range(1, n + 1):
                        c = i * (inc // n)
                        m = (
                            nn.Conv2d(
                                c,
                                c,
                                kernel_size=k,
                                stride=s,
                                padding=p,
                                groups=c,
                                bias=True if bias == 1 else False,
                            )
                            .cuda()
                            .eval()
                        )
                        l = measure_net_latency(m, input_shape=(c, r, r))[0]
                        config["records"].append((c, c, l))
                else:
                    i_n, o_n = make_divisor(inc), make_divisor(outc)
                    for i in range(1, i_n + 1):
                        for j in range(1, o_n + 1):
                            ic, oc = i * (inc // i_n), j * (outc // o_n)
                            m = (
                                nn.Conv2d(
                                    ic,
                                    oc,
                                    kernel_size=k,
                                    stride=s,
                                    padding=p,
                                    groups=1,
                                    bias=True if bias == 1 else False,
                                )
                                .cuda()
                                .eval()
                            )
                            l = measure_net_latency(m, input_shape=(ic, r, r))[0]
                            config["records"].append((ic, oc, l))
            elif args.startswith("linear") and True:
                inc, outc = config["max_in"], config["max_out"]
                assert outc == 1000
                n = make_divisor(inc)
                for i in range(1, n + 1):
                    c = i * (inc // n)
                    m = nn.Linear(c, outc).cuda().eval()
                    l = measure_net_latency(m, input_shape=(c,))[0]
                    config["records"].append((c, outc, l))
            else:
                module = (
                    {
                        "silu": lambda: nn.SiLU(inplace=True),
                        "sigmoid": nn.Sigmoid,
                        "pool": lambda: SelectAdaptivePool2d(
                            pool_type="avg", flatten=True
                        ),
                        "add": AddModule,
                        "mult": MultModule,
                    }[args.split("_")[0]]()
                    .cuda()
                    .eval()
                )
                r = int(re.findall(r"r(\d+)", args)[0])
                max_c = max(config["max_in"], config["max_out"])
                n = make_divisor(max_c)
                for i in range(1, n + 1):
                    c = i * (max_c // n)
                    l = measure_net_latency(module, input_shape=(c, r, r))[0]
                    config["records"].append((c, c, l))
        print(json.dump(basic_modules, open("l.json", "w"), indent=4))

    def test_predict(self):
        replaece_InvertedResidual_forward()
        replaece_Mult_forward()

        model: EfficientNet = efficientnet_b4().cuda().eval()
        print(measure_net_latency(model)[0])
        collect_info(model)

        predictor = Predictor()
        predictor.build_net()
        print(predictor.predict(model))

    def test_build_predictor(self):
        predictor = Predictor()
        predictor.build_net()

    def test_predict_one_module(self):
        replaece_InvertedResidual_forward()
        replaece_Mult_forward()

        model: EfficientNet = efficientnet_b0().cuda().eval()
        print(measure_net_latency(model)[0])

        predictor = Predictor()
        predictor.build_net()

        # collec info
        collect_info(model)

        print(predictor.nets["linear"].forward(1792, 1000))
        print(predictor.nets["linear"].forward(32, 1000))
        print(predictor.nets["linear"].forward(544, 1000))
        print(predictor.nets["linear"].forward(560, 1000))
        print(predictor.nets["linear"].forward(576, 1000))

        for args in predictor.nets:
            print(args, type(predictor.nets[args]))

    def test_predict_one_module_2d(self):

        predictor = Predictor()
        predictor.build_net()

        print(predictor.nets["conv2d_k1_s1_p0_r1_b1_dw0"].forward(48, 1152))

        print(predictor.nets["conv2d_k1_s1_p0_r1_b1_dw0"].forward(128, 1152))
        print(predictor.nets["conv2d_k1_s1_p0_r1_b1_dw0"].forward(24, 72))
        print(predictor.nets["conv2d_k1_s1_p0_r1_b1_dw0"].forward(25, 70))
        print(predictor.nets["conv2d_k1_s1_p0_r1_b1_dw0"].forward(25, 72))

        print(predictor.nets["conv2d_k1_s1_p0_r1_b1_dw0"].forward(25, 71))

    def test_algo(self):
        replaece_InvertedResidual_forward()
        replaece_Mult_forward()
        model = efficientnet_b4()
        algo = EffDmsAlgorithm.init_fold_algo(
            model, mutator_kwargs={"type": "EffLatencyMutator"}
        )
        print(algo.mutator.info())

        static = torch.load("last_latency.pth.tar", map_location="cpu")["state_dict"]
        algo.load_state_dict(static, strict=False)
        static = algo.to_static_model().eval().cuda()

        print(measure_net_latency(static, device=torch.device("cpu"))[0])
        print(measure_net_latency(static, device=torch.device("cuda:0"))[0])

        print(
            ptflops.get_model_complexity_info(
                static, input_res=(3, 224, 224), print_per_layer_stat=False
            )
        )

        # print(static)
        algo2 = EffDmsAlgorithm(static)
        print(algo2.mutator.info())

    def test_grad(self):
        a = nn.Parameter(torch.tensor([100.0]))
        b = nn.Parameter(torch.tensor([1000.0]))

        p = Predictor()
        p.build_net()

        loss = p.nets["linear"](a, b)
        loss.backward()
        print(a.grad)
        print(b.grad)

    def test_latency(self):

        model = efficientnet_b4().eval().cuda()
        algo = EffDmsAlgorithm(model)
        static = torch.load("last.pth.tar", map_location="cpu")["state_dict"]
        algo.load_state_dict(static)

        static = algo.to_static_model().eval().cuda()
        print(measure_net_latency(static)[0])
        # print(measure_net_latency(static, device=torch.device("cpu"))[0])
        print(
            ptflops.get_model_complexity_info(
                static, (3, 224, 224), print_per_layer_stat=False
            )[0]
        )
        print(thop.profile(static, [torch.ones([1, 3, 224, 224]).cuda()])[0])

        model = efficientnet_b0().eval().cuda()
        print(measure_net_latency(model)[0])
        # print(measure_net_latency(model, device=torch.device("cpu"))[0])
        print(
            ptflops.get_model_complexity_info(
                model, (3, 224, 224), print_per_layer_stat=False
            )[0]
        )
        print(thop.profile(model, [torch.ones([1, 3, 224, 224]).cuda()])[0])

        print()

        model = efficientnet_b1().eval().cuda()
        print(measure_net_latency(model)[0])
        # print(measure_net_latency(model, device=torch.device("cpu"))[0])
        print(
            ptflops.get_model_complexity_info(
                model, (3, 224, 224), print_per_layer_stat=False
            )[0]
        )
        print(thop.profile(model, [torch.ones([1, 3, 224, 224]).cuda()])[0])

        print()

    def test_smooth(self):
        import scipy
        import numpy as np
        from scipy.signal import savgol_filter

        # np.set_printoptions(precision=2)  # For compact display.
        x = np.array([2, 2, 5, 2, 1, 0, 1, 4, 9])
        savgol_filter(x, 5, 2)

        data = json.load(open("l.json"))["add_r14"]
        data = [r for _, _, r in data["records"]]
        print(data)
        print(savgol_filter(data, 5, 2))

    def test_split2(self):
        replaece_InvertedResidual_forward()
        replaece_Mult_forward()

        model: EfficientNet = efficientnet_b0().cuda().eval()
        print(measure_net_latency(model)[0])

        collect_info(model)

        # meaure
        predictor = Predictor()

        quick_measure = lambda m: measure_net_latency(m, input_shape=m.in_shape[1:])[0]

        for n, m in model.named_modules():
            if isBasicModule(m) and hasattr(m, "in_shape"):
                l = quick_measure(m)
                print(n, m, l, predictor.parse_m(m))

        print(measure_net_latency(model)[0])
