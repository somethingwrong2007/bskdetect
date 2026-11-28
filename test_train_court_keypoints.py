import json
from types import SimpleNamespace

import cv2
import numpy as np

from bsk.perceive import court_keypoints as ck
from bsk.perceive.court_keypoints import KEYPOINT_IDS
from bsk.train_court_keypoints import (
    INPUT_HW,
    SOURCE_KEYPOINTS,
    SOURCE_TO_TARGET,
    _fit_label_homography,
    _render_synthetic_court,
    assign_video_groups,
    augment,
    build_torch,
    checkpoint_score,
    convert_split,
    decode_heatmaps,
    evaluate,
    export,
    load_split,
    loss_fn,
    make_targets,
    video_group,
)


def _write_source(root):
    root.mkdir(parents=True)
    image = np.zeros((32, 48, 3), np.uint8)
    assert cv2.imwrite(str(root / "court.jpg"), image)
    points = np.zeros((len(SOURCE_KEYPOINTS), 3), np.float32)
    for index in range(len(SOURCE_KEYPOINTS)):
        points[index] = ((index * 1.3) % 46 + 1, (index * 2.7) % 30 + 1, 2)
    data = {
        "images": [{"id": 1, "file_name": "court.jpg", "width": 48, "height": 32}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [0, 0, 48, 32], "area": 1536, "iscrowd": 0,
                         "keypoints": points.reshape(-1).tolist()}],
        "categories": [{"id": 0, "name": "basketball-court"},
                       {"id": 1, "name": "court", "keypoints": list(SOURCE_KEYPOINTS)}],
    }
    (root / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def _metrics(**values):
    result = {
        "gated_pck10": 0.0,
        "homography_accurate_rate": 0.0,
        "homography_false_publish_rate": 0.0,
        "absent_fpr": 0.0,
    }
    result.update(values)
    return result


def test_export_nonzero_probe_matches_pytorch_and_supports_dynamic_batch(tmp_path):
    torch = __import__("torch")
    CourtNet, _ = build_torch("tiny", pretrained=False)
    model = CourtNet().eval()
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"model": model.state_dict(), "architecture": "tiny",
                "keypoint_ids": list(KEYPOINT_IDS), "input_hw": list(INPUT_HW)}, checkpoint)
    output = tmp_path / "tiny.onnx"

    sidecar = export(SimpleNamespace(checkpoint=checkpoint, output=output))

    assert output.is_file()
    assert sidecar["export_parity"]["coordinate_max_abs_diff_px"] <= 0.1
    assert sidecar["export_parity"]["confidence_max_abs_diff"] <= 1e-3
    import onnxruntime as ort
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    for batch in (1, 2):
        result = session.run(None, {"input": np.zeros((batch, 3, *INPUT_HW), np.float32)})[0]
        assert result.shape == (batch, len(KEYPOINT_IDS), 3)
        assert np.isfinite(result).all()


def test_checkpoint_score_prefers_accurate_geometry_over_false_publication():
    # Both candidates have identical real PCK, but only one reconstructs the
    # complete court accurately on the independent rendered validation split.
    real = _metrics(gated_pck10=0.65)
    accurate = _metrics(
        gated_pck10=0.75,
        homography_accurate_rate=0.80,
        homography_false_publish_rate=0.05,
    )
    false_publication = _metrics(
        gated_pck10=0.75,
        homography_accurate_rate=0.0,
        homography_false_publish_rate=1.0,
    )

    assert checkpoint_score(real, accurate) > checkpoint_score(real, false_publication)
    assert checkpoint_score(real, accurate) > checkpoint_score(real)
    false_real_h = _metrics(
        gated_pck10=0.70,
        homography_false_publish_rate=1.0,
    )
    clean_real_h = _metrics(gated_pck10=0.65)
    assert checkpoint_score(clean_real_h, accurate) > checkpoint_score(false_real_h, accurate)


def test_synthetic_court_image_and_visible_coco_labels_contract():
    # Cover all three render styles, including deterministic partial occlusion.
    for seed in (11, 1, 5):
        image, points = _render_synthetic_court(np.random.default_rng(seed))
        assert image.shape == (640, 640, 3)
        assert image.dtype == np.uint8
        assert points.shape == (len(KEYPOINT_IDS), 3)
        assert points.dtype == np.float32
        assert np.isfinite(points).all()
        assert set(np.unique(points[:, 2])) <= {0.0, 2.0}
        visible = points[:, 2] == 2.0
        assert visible.any()
        assert ((points[visible, 0] >= 0) & (points[visible, 0] < INPUT_HW[1])).all()
        assert ((points[visible, 1] >= 0) & (points[visible, 1] < INPUT_HW[0])).all()


def test_label_homography_rejects_four_collinear_world_points():
    indices = [0, 1, 12, 13]  # Four distinct landmarks on the same baseline.
    world = ck.WORLD_POINTS[indices]
    assert np.linalg.matrix_rank(world - world.mean(axis=0)) == 1
    points = np.zeros((len(KEYPOINT_IDS), 3), np.float32)
    points[indices, :2] = ((100, 100), (500, 120), (180, 420), (520, 480))
    points[indices, 2] = 2.0

    matrix, errors = _fit_label_homography(points)

    assert matrix is None
    assert errors.size == 0


def test_label_homography_accepts_exact_noncollinear_four_corners():
    indices = [0, 1, 2, 3]
    H_inv = np.asarray([[22.0, 0.0, 320.0],
                        [0.0, 22.0, 320.0],
                        [0.0, 0.0, 1.0]], np.float64)
    pixels = ck._project(H_inv, ck.WORLD_POINTS[indices])
    points = np.zeros((len(KEYPOINT_IDS), 3), np.float32)
    points[indices, :2] = pixels
    points[indices, 2] = 2.0

    matrix, errors = _fit_label_homography(points)

    assert matrix is not None and np.isfinite(matrix).all()
    assert errors.shape == (4,)
    np.testing.assert_allclose(errors, 0.0, atol=1e-4)
    np.testing.assert_allclose(ck._project(matrix, ck.WORLD_POINTS[indices]), pixels, atol=1e-4)


def test_convert_selects_exact_runtime_order_and_clears_offscreen(tmp_path):
    source = tmp_path / "source"
    data = _write_source(source)
    # Point 41 is the last source landmark. Move it outside the image while
    # retaining a visible flag to verify the safety cleanup.
    raw = np.asarray(data["annotations"][0]["keypoints"]).reshape(-1, 3)
    raw[-1] = (99, 99, 2)
    data["annotations"][0]["keypoints"] = raw.reshape(-1).tolist()
    (source / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")

    destination = tmp_path / "converted"
    summary = convert_split(source, destination)
    converted = json.loads((destination / "_annotations.coco.json").read_text(encoding="utf-8"))
    category = converted["categories"][0]
    assert tuple(category["keypoints"]) == KEYPOINT_IDS
    result = np.asarray(converted["annotations"][0]["keypoints"]).reshape(-1, 3)
    source_index = {name: index for index, name in enumerate(SOURCE_KEYPOINTS)}
    for target_index, target_name in enumerate(KEYPOINT_IDS):
        expected = raw[source_index[SOURCE_TO_TARGET[target_name]]]
        if target_name == "corner_A_far":
            expected = np.array((0, 0, 0), np.float32)
        assert np.array_equal(result[target_index], expected)
    # corner_A_far is source 41 and must be marked absent after going off-screen.
    assert np.array_equal(result[KEYPOINT_IDS.index("corner_A_far")], (0, 0, 0))
    assert summary["images"] == summary["annotations"] == 1
    assert len(load_split(tmp_path / "converted", ".")) == 1


def test_augmentation_preserves_visibility_layout():
    image = np.full((64, 64, 3), 127, np.uint8)
    points = np.zeros((16, 3), np.float32)
    points[:4] = ((5, 5, 2), (58, 5, 2), (58, 58, 2), (5, 58, 2))
    visibility_before = points[:, 2].copy()
    augmented, result = augment(image, points, np.random.default_rng(7))
    assert augmented.shape == image.shape
    assert result.shape == (16, 3)
    assert np.all(result[:, 2] <= visibility_before)
    assert np.all(result[visibility_before == 0, 2] == 0)
    assert np.isfinite(result[:, :2]).all()


def test_decode_heatmaps_preserves_symmetric_subpixel_center():
    torch = __import__("torch")
    heatmaps = torch.full((1, 1, 8, 8), -100.0)
    # Equal adjacent peaks are symmetric about x=3.5. A hard argmax would
    # incorrectly snap to x=3 instead of decoding the half-pixel center.
    heatmaps[0, 0, 3, 3:5] = 10.0

    points, confidence = decode_heatmaps(heatmaps, torch)

    torch.testing.assert_close(points, torch.tensor([[[7.0, 6.0]]]), atol=1e-6, rtol=0)
    assert confidence.shape == (1, 1)
    assert torch.isfinite(confidence).all()


def test_export_applies_peak_sigmoid_exactly_once():
    torch = __import__("torch")

    class FixedModel(torch.nn.Module):
        def forward(self, _images):
            heatmaps = torch.full((1, 1, 8, 8), -2.0)
            presence = torch.zeros(1, 1)
            return heatmaps, presence

    _CourtNet, ExportNet = build_torch()
    output = ExportNet(FixedModel()).eval()(torch.zeros(1, 3, 8, 8))
    expected = 0.5 * torch.sigmoid(torch.tensor(-2.0))

    torch.testing.assert_close(output[0, 0, 2], expected)


def test_resnet18_constructs_without_pretrained_download(monkeypatch):
    from torchvision import models

    original_resnet18 = models.resnet18

    def offline_resnet18(*args, **kwargs):
        assert kwargs.get("weights") is None
        return original_resnet18(*args, **kwargs)

    monkeypatch.setattr(models, "resnet18", offline_resnet18)
    CourtNet, _ExportNet = build_torch("resnet18", pretrained=False)
    model = CourtNet().eval()

    assert model.stem[0].out_channels == 64
    assert model.enc4[-1].conv2.out_channels == 512
    assert model.heatmap_head.out_channels == len(KEYPOINT_IDS)
    assert model.presence_head[-1].out_features == len(KEYPOINT_IDS)


def test_resnet18_uint8_batch_normalization_is_finite():
    torch = __import__("torch")
    CourtNet, _ = build_torch("resnet18", pretrained=False)
    model = CourtNet().eval()
    images = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)

    with torch.no_grad():
        heatmaps, presence = model(images)

    assert torch.isfinite(heatmaps).all()
    assert torch.isfinite(presence).all()
    assert heatmaps.shape[0] == presence.shape[0] == 2


def test_source_video_grouping_prevents_cross_split_leakage():
    name = ("denver-nuggets-los-angeles-clippers-game-1-q1-12_00-11_35_mp4-0002_"
            "jpg.rf.572060285be93e26d99c6b18fc989852.jpg")
    assert video_group(name) == "denver-nuggets-los-angeles-clippers-game-1"
    groups = {f"game-{index:02d}": list(range(index + 1)) for index in range(18)}
    assignment = assign_video_groups(groups, seed=17)
    flattened = [name for names in assignment.values() for name in names]
    assert len(flattened) == len(set(flattened)) == len(groups)
    assert set(flattened) == set(groups)
    assert [len(assignment[split]) for split in ("train", "valid", "test")] == [12, 3, 3]


def test_targets_and_export_wrapper_contract():
    torch = __import__("torch")
    CourtNet, ExportNet = build_torch()
    model = CourtNet().eval()
    # A small input keeps this unit test fast; spatial sizes are dynamic. The
    # real export is separately checked at the fixed 640x640 runtime input.
    output = ExportNet(model)(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 16, 3)
    assert torch.isfinite(output).all()
    assert ((output[..., :2] >= 0) & (output[..., :2] < 64)).all()
    assert ((output[..., 2] >= 0) & (output[..., 2] <= 1)).all()

    points = torch.zeros(1, 16, 2)
    visibility = torch.zeros(1, 16)
    visibility[0, 0] = 2.0
    targets, target_visibility = make_targets(points, visibility, torch)
    assert targets.shape == (1, 16, 320, 320)
    assert float(targets.max()) == 1.0
    assert torch.equal(target_visibility, (visibility > 0).float())
    total, heat, presence = loss_fn(
        torch.zeros(1, 16, 320, 320), torch.zeros(1, 16), targets, target_visibility, torch)
    assert float(total) >= 0.0 and float(heat) >= 0.0 and float(presence) >= 0.0


def test_evaluate_counts_keypoints_not_pixel_distances():
    torch = __import__("torch")

    class FixedModel:
        def eval(self):
            return self

        def __call__(self, images):
            batch = images.shape[0]
            heatmaps = torch.zeros(batch, 16, 320, 320)
            presence = torch.full((batch, 16), -2.0)
            presence[:, :2] = 2.0
            return heatmaps, presence

    images = torch.zeros(1, 3, 640, 640, dtype=torch.uint8)
    points = torch.zeros(1, 16, 2)
    points[0, 1, 0] = 10.0
    visibility = torch.zeros(1, 16)
    visibility[0, :2] = 1.0
    metrics = evaluate(FixedModel(), [(images, points, visibility,
                                      torch.tensor([[640, 640]], dtype=torch.int32))],
                       torch.device("cpu"), torch)
    assert metrics["visible"] == 2
    assert metrics["pck5"] == 0.5
    assert metrics["pck10"] == 1.0
    assert metrics["pck20"] == 1.0
    assert metrics["gated_pck5"] == 0.5
    assert metrics["gated_pck10"] == 1.0
    assert metrics["presence_recall"] == 1.0
    assert metrics["images"] == 1
    assert metrics["homography_publish_rate"] == 0.0
    assert 0.0 <= metrics["mean_conf"] <= 1.0
    assert 0.0 <= metrics["absent_fpr"] <= 1.0


def test_evaluate_publishes_homography_with_world_coordinates():
    torch = __import__("torch")
    H_inv = np.asarray([[16.0, 0.0, 288.0],
                        [0.0, 16.0, 162.0],
                        [0.00005, 0.0, 1.0]], np.float64)
    source_height, source_width = 324, 576
    pixels = ck._project(H_inv, ck.WORLD_POINTS)
    pixels[:, 0] *= 640.0 / source_width
    pixels[:, 1] *= 640.0 / source_height
    peak_index = np.rint(pixels / 2.0).astype(int)

    class ExactModel:
        def eval(self):
            return self

        def __call__(self, images):
            heatmaps = torch.full((1, 16, 320, 320), -10.0)
            for keypoint, (x, y) in enumerate(peak_index):
                heatmaps[0, keypoint, y, x] = 10.0
            return heatmaps, torch.full((1, 16), 5.0)

    images = torch.zeros(1, 3, 640, 640, dtype=torch.uint8)
    points = torch.from_numpy(pixels.astype(np.float32))[None]
    visibility = torch.ones(1, 16)
    source_hw = torch.tensor([[source_height, source_width]], dtype=torch.int32)
    metrics = evaluate(ExactModel(), [(images, points, visibility, source_hw)],
                       torch.device("cpu"), torch)
    assert metrics["pck10"] == 1.0
    assert metrics["gated_pck10"] == 1.0
    assert metrics["homography_publish_rate"] == 1.0
    assert metrics["homography_reproj_median_px"] < 1.0
    assert metrics["homography_false_publish_rate"] == 0.0
    assert metrics["homography_evaluated_accuracy"] == 1.0

    # The model can publish all 16 points while only three human labels exist.
    # That H is unevaluable, not a demonstrated false publication.
    sparse_visibility = torch.zeros(1, 16)
    sparse_visibility[0, :3] = 1.0
    sparse_metrics = evaluate(ExactModel(), [(images, points, sparse_visibility, source_hw)],
                              torch.device("cpu"), torch)
    assert sparse_metrics["homography_publish_rate"] == 1.0
    assert sparse_metrics["homography_accurate_rate"] == 0.0
    assert sparse_metrics["homography_false_publish_rate"] == 0.0
