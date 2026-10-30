#!/usr/bin/env python3
"""학습 환경 확인: GPU/torch(sm_120), mmcv CUDA 연산, RTMDet-Ins 6클래스 모델 bf16 학습 1스텝, 메모리.

원본 해상도 조각 학습을 가정해 side×side 무작위 이미지 + 작은 물체 정답으로 forward/backward 를 돌리고
최대 GPU 메모리를 출력한다. 배치를 늘려 가며 16 GB 안에 들어가는 크기를 고르면 된다.
예) python check_env.py --config model_config.py --side 1024 --batch 2 4
"""
import argparse
import time

import numpy as np
import torch


def check_basics():
    print("torch", torch.__version__, "cuda", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0),
          "| capability", torch.cuda.get_device_capability(0), "| arch", torch.cuda.get_arch_list())
    import mmcv
    import mmdet
    import mmengine
    print("mmcv", mmcv.__version__, "mmdet", mmdet.__version__, "mmengine", mmengine.__version__)
    from mmcv.ops import nms, roi_align
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float32, device="cuda")
    keep = nms(boxes, torch.tensor([0.9, 0.8, 0.7], device="cuda"), 0.5)[1]
    assert keep.tolist() == [0, 2], keep
    feat = torch.randn(1, 4, 32, 32, device="cuda")
    roi_align(feat, torch.tensor([[0, 2, 2, 20, 20]], dtype=torch.float32, device="cuda"), (7, 7))
    print("mmcv CUDA ops OK (nms, roi_align)")


def train_step(config, side, batch, steps=3):
    from mmengine.config import Config
    from mmdet.registry import MODELS
    from mmdet.structures import DetDataSample
    from mmdet.structures.mask import BitmapMasks
    from mmengine.structures import InstanceData
    from mmdet.utils import register_all_modules
    register_all_modules()
    cfg = Config.fromfile(config)
    cfg.model.bbox_head.num_classes = 6
    model = MODELS.build(cfg.model).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(0)

    def sample():
        n = 20
        masks = np.zeros((n, side, side), np.uint8)
        boxes = []
        for k in range(n):
            s = int(rng.integers(4, 40))  # 원본 해상도 기준 4~40 px 물체
            x, y = int(rng.integers(0, side - s)), int(rng.integers(0, side - s))
            masks[k, y:y + s, x:x + s] = 1
            boxes.append([x, y, x + s, y + s])
        ds = DetDataSample(metainfo=dict(img_shape=(side, side), pad_shape=(side, side), ori_shape=(side, side),
                                         scale_factor=(1.0, 1.0), batch_input_shape=(side, side)))
        ds.gt_instances = InstanceData(bboxes=torch.tensor(boxes, dtype=torch.float32),
                                       labels=torch.tensor(rng.integers(0, 6, n)),
                                       masks=BitmapMasks(masks, side, side))
        return ds

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        data = dict(inputs=[torch.randint(0, 255, (3, side, side), dtype=torch.uint8) for _ in range(batch)],
                    data_samples=[sample() for _ in range(batch)])
        data = model.data_preprocessor(data, training=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model._run_forward(data, mode="loss")
            loss, _ = model.parse_losses(losses)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"side {side} batch {batch}: loss {float(loss.detach()):.3f}, {(time.time() - t0) / steps:.2f} s/step, "
          f"peak {peak:.1f} / {total:.1f} GiB")
    del model, opt
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--side", type=int, default=1024)
    ap.add_argument("--batch", type=int, nargs="+", default=[2, 4])
    args = ap.parse_args()
    check_basics()
    for b in args.batch:
        try:
            train_step(args.config, args.side, b)
        except torch.cuda.OutOfMemoryError:
            print(f"side {args.side} batch {b}: 메모리 부족")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
