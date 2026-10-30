"""mmdet Resize 의 마스크 축소를 픽셀 중심 정렬 방식으로 바꾸는 패치.

문제: mmdet BitmapMasks.rescale/resize 는 cv2.INTER_NEAREST 로 마스크를 줄인다. OpenCV 최근접은
출력 픽셀 d 에 원본 floor(d*s) 를 쓰기 때문에(픽셀 중심 비정렬) 2460→1600 축소에서 정답 마스크가
원본 기준 오른쪽·아래로 약 0.7 px 밀린다. 이미지는 INTER_LINEAR(중심 정렬)로 줄어들어 서로 어긋난다.
모델은 밀린 정답을 배우므로 예측도 +0.7~1 px 밀리고, 4~12 px 물체의 IoU 가 0.5 아래로 떨어진다.

해결: 마스크를 float 로 INTER_AREA 축소(중심 정렬, 면적 평균) 후 0.5 로 이진화.
사용: 학습 스크립트에서 mmdet 를 import 한 뒤, 데이터 파이프라인을 만들기 전에
    import mask_resize_fix; mask_resize_fix.install()
확인: python3 mask_resize_fix.py  (패치 전후 밀림을 출력)
"""
import cv2
import numpy as np


def _resize_masks(masks, out_h, out_w):
    if len(masks) == 0:
        return np.empty((0, out_h, out_w), np.uint8)
    out = []
    for m in masks:
        a = cv2.resize(m.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA)
        b = a >= 0.5
        if m.any() and not b.any():  # 아주 작은 물체가 사라지지 않게 가장 많이 덮인 픽셀은 남김
            b[np.unravel_index(int(a.argmax()), a.shape)] = True
        out.append(b)
    return np.stack(out).astype(np.uint8)


def install():
    import mmcv
    from mmdet.structures.mask import BitmapMasks

    def rescale(self, scale, interpolation='nearest'):
        if len(self.masks) == 0:
            new_w, new_h = mmcv.rescale_size((self.width, self.height), scale)
            return BitmapMasks(np.empty((0, new_h, new_w), np.uint8), new_h, new_w)
        new_h, new_w = mmcv.imrescale(self.masks[0], scale, interpolation='nearest').shape[:2]  # 크기 계산은 기존과 동일
        return BitmapMasks(_resize_masks(self.masks, new_h, new_w), new_h, new_w)

    def resize(self, out_shape, interpolation='nearest'):
        h, w = out_shape
        return BitmapMasks(_resize_masks(self.masks, h, w), h, w)

    BitmapMasks.rescale = rescale
    BitmapMasks.resize = resize
    BitmapMasks._center_aligned_resize = True


def measure(n=300, H=1848, W=2460, side=1600, seed=0):
    """작은 원 마스크를 축소→bilinear 복원했을 때 원본 대비 중심 밀림(px)과 면적비."""
    import torch
    import torch.nn.functional as F
    from mmdet.structures.mask import BitmapMasks
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n):
        m = np.zeros((H, W), np.uint8)
        cv2.circle(m, (int(rng.uniform(100, W - 100)), int(rng.uniform(100, H - 100))), int(rng.uniform(3, 15)), 1, -1)
        small = BitmapMasks(m[None], H, W).rescale((side, side)).masks[0].astype(np.float32)
        up = F.interpolate(torch.from_numpy(small)[None, None], size=(H, W), mode='bilinear',
                           align_corners=False)[0, 0].numpy() > 0.5
        ys, xs = np.nonzero(m)
        ys2, xs2 = np.nonzero(up)
        if len(xs2):
            res.append((xs2.mean() - xs.mean(), ys2.mean() - ys.mean(), up.sum() / m.sum()))
    r = np.array(res)
    return dict(dx=round(float(r[:, 0].mean()), 2), dy=round(float(r[:, 1].mean()), 2),
                area_ratio=round(float(np.median(r[:, 2])), 2), lost=n - len(r))


if __name__ == '__main__':
    print('before (mmdet nearest):', measure())
    install()
    print('after  (center-aligned):', measure())
