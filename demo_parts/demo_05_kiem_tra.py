"""Kiem thu so va tao metrics.json sau khi chay cac phan demo."""

import hashlib
import json
import sys

import numpy as np
import scipy
from scipy import ndimage as ndi
import skimage as ski
import matplotlib
import PIL

from demo_common import OUT, K_LAP
from demo_03_phat_hien_bien import non_maximum_suppression, zero_crossings
from demo_04_pyramid_blending import gaussian_pyramid, laplacian_pyramid, reconstruct


def main():
    checks = {}
    ramp = np.tile(np.arange(9.0), (7, 1))
    kx = np.array([[1.0, -1.0]])
    ky = kx.T
    assert np.allclose(ndi.convolve(ramp, kx, mode="reflect")[:, :-1], 1)
    assert np.allclose(ndi.convolve(ramp, ky, mode="reflect"), 0)
    assert np.allclose(ndi.convolve(np.ones((7, 9)), K_LAP, mode="reflect"), 0)
    checks["forward_difference_and_constant_laplacian"] = True

    for degree, neighbor in [(45, (3, 3)), (135, (3, 1)), (0, (2, 3)), (90, (3, 2))]:
        magnitude = np.zeros((5, 5))
        magnitude[2, 2] = 2
        magnitude[neighbor] = 3
        assert non_maximum_suppression(magnitude, np.full_like(magnitude, np.deg2rad(degree)))[2, 2] == 0
    checks["nms_four_directions"] = True

    camera = ski.img_as_float(ski.data.camera())
    kx_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 4
    ky_sobel = kx_sobel.T
    gx = ndi.correlate(camera, kx_sobel, mode="reflect")
    gy = ndi.correlate(camera, ky_sobel, mode="reflect")
    assert np.allclose(gx, ski.filters.sobel_v(camera))
    assert np.allclose(gy, ski.filters.sobel_h(camera))
    assert np.allclose(ski.filters.sobel(camera), np.hypot(gx, gy) / np.sqrt(2))
    checks["sobel_axes_sign_and_normalization"] = True

    errors = {}
    rng = np.random.default_rng(42)
    for shape in [(65, 79), (65, 79, 3), (64, 64, 3)]:
        test_image = rng.random(shape)
        error = float(np.max(np.abs(reconstruct(laplacian_pyramid(gaussian_pyramid(test_image, 4))) - test_image)))
        assert error < 1e-12
        errors[str(shape)] = error
    checks["pyramid_odd_even_gray_rgb"] = errors

    image_a = ski.transform.resize(ski.img_as_float(ski.data.astronaut()), (512, 512), anti_aliasing=True)
    image_b = ski.transform.resize(ski.img_as_float(ski.data.coffee()), (512, 512), anti_aliasing=True)
    for value, target in [(1, image_a), (0, image_b)]:
        mask = np.full_like(image_a, value)
        gaussian_mask = gaussian_pyramid(mask, 5)
        lap_a = laplacian_pyramid(gaussian_pyramid(image_a, 5))
        lap_b = laplacian_pyramid(gaussian_pyramid(image_b, 5))
        mixed = [m * a + (1 - m) * b for m, a, b in zip(gaussian_mask, lap_a, lap_b)]
        assert np.max(np.abs(reconstruct(mixed) - target)) < 1e-12
    checks["blend_mask_endpoints"] = True

    assert not zero_crossings(np.ones((7, 7))).any()
    checks["zero_crossing_constant_image"] = True

    canny1 = ski.feature.canny(camera, sigma=1, low_threshold=0.1, high_threshold=0.2)
    canny3 = ski.feature.canny(camera, sigma=3, low_threshold=0.1, high_threshold=0.2)
    pyramid_shapes = [list(level.shape) for level in gaussian_pyramid(
        ski.transform.resize(ski.img_as_float(ski.data.astronaut()), (512, 512), anti_aliasing=True), 5)]
    base = ski.transform.resize(ski.img_as_float(ski.data.astronaut()), (512, 512), anti_aliasing=True)
    reconstruction_max_abs_error = float(np.max(np.abs(reconstruct(laplacian_pyramid(gaussian_pyramid(base, 5))) - base)))
    figures = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(OUT.glob("*.png"))}
    metrics = {
        "python": sys.version.split()[0],
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit-image": ski.__version__, "matplotlib": matplotlib.__version__, "pillow": PIL.__version__},
        "reconstruction_max_abs_error": reconstruction_max_abs_error,
        "canny_edge_pixels": {"sigma1": int(canny1.sum()), "sigma3": int(canny3.sum())},
        "gaussian_shapes": pyramid_shapes,
        "checks": checks,
        "figures": figures,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Tat ca kiem tra so da dat.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
