import torch
from krea2_detailer import _resolve_detail_mask


def _box(h, w, x0, y0, x1, y1):
    m = torch.zeros(h, w)
    m[y0:y1, x0:x1] = 1.0
    return m


def main():
    H, W = 1024, 1024
    small = _box(H, W, 400, 200, 620, 760)
    large = _box(H, W, 180, 80, 760, 980)
    huge = _box(H, W, 0, 0, 1024, 1024)

    m1, s1 = _resolve_detail_mask(small, 'auto', H * W)
    assert s1 == 'full' and m1 is not None

    m2, s2 = _resolve_detail_mask(large, 'auto', H * W)
    assert s2 == 'head' and m2 is not None
    assert (m2 > 0.5).sum().item() < (large > 0.5).sum().item()

    m3, s3 = _resolve_detail_mask(huge, 'auto', H * W)
    assert m3 is None and s3 == 'skip'
    print('ok')


if __name__ == '__main__':
    main()
