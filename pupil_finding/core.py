from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def odd_kernel_size(value: float, minimum: int = 3) -> int:
    value = int(round(value))
    value = max(minimum, value)
    if value % 2 == 0:
        value += 1
    return value


def full_image_scale_params(gray_shape: tuple[int, int]) -> dict[str, float | int]:
    height, width = gray_shape
    image_area = height * width
    short_side = min(height, width)
    return {
        "full_blur_sigma": max(2.0, short_side * 0.0064),
        "full_blackhat_kernel": odd_kernel_size(short_side * 0.082, minimum=31),
        "seed_area_min": max(20, int(image_area * 0.00012)),
        "seed_area_max": max(80, int(image_area * 0.0028)),
        "fallback_area_min": max(80, int(image_area * 0.0006)),
        "fallback_area_max": max(600, int(image_area * 0.013)),
        "seed_min_roi_w": max(45, int(short_side * 0.11)),
        "seed_min_roi_h": max(40, int(short_side * 0.095)),
    }


def roi_scale_params(roi_shape: tuple[int, int]) -> dict[str, float | int | None]:
    height, width = roi_shape
    roi_area = height * width
    short_side = min(height, width)
    return {
        "roi_blur_sigma": max(1.0, short_side * 0.025),
        "pupil_blackhat_kernel": odd_kernel_size(short_side * 0.32, minimum=15),
        "pupil_area_min": max(8, int(roi_area * 0.003)),
        "pupil_area_max": max(80, int(roi_area * 0.14)),
        "glint_threshold": None,
    }


def fit_circle_least_squares(points: np.ndarray) -> tuple[float, float, float]:
    points = points.reshape(-1, 2).astype(np.float64)
    x = points[:, 0]
    y = points[:, 1]
    matrix = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    target = x**2 + y**2
    coeffs, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    cx, cy, d = coeffs
    radius = np.sqrt(cx**2 + cy**2 + d)
    return float(cx), float(cy), float(radius)


def circle_fit_residual(points: np.ndarray, cx: float, cy: float, radius: float) -> float:
    pts = points.reshape(-1, 2).astype(np.float64)
    radial = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return float(np.mean(np.abs(radial - radius)))


def expand_bbox(
    x: int,
    y: int,
    width: int,
    height: int,
    img_shape: tuple[int, int],
    scale_x: float = 2.3,
    scale_y: float = 2.0,
) -> tuple[int, int, int, int]:
    image_h, image_w = img_shape
    cx = x + width / 2
    cy = y + height / 2
    new_w = int(width * scale_x)
    new_h = int(height * scale_y)
    new_x = int(cx - new_w * 0.58)
    new_y = int(cy - new_h * 0.55)
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    new_w = min(new_w, image_w - new_x)
    new_h = min(new_h, image_h - new_y)
    return new_x, new_y, new_w, new_h


def estimate_eye_fissure_features(roi_gray: np.ndarray) -> dict[str, float]:
    height, width = roi_gray.shape
    if height < 10 or width < 10:
        return _empty_fissure()

    sigma = max(1.0, min(height, width) * 0.025)
    blurred = cv2.GaussianBlur(roi_gray, (0, 0), sigmaX=sigma)
    _, dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    )
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: dict[str, float] | None = None
    roi_area = height * width
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < roi_area * 0.015 or area > roi_area * 0.75:
            continue
        _, _, bbox_w, bbox_h = cv2.boundingRect(contour)
        aspect = bbox_w / max(1, bbox_h)
        area_ratio = area / roi_area
        horizontal = min(1.0, aspect / 2.2)
        score = min(area_ratio / 0.28, 1.0) * min(aspect / 2.2, 1.4) * horizontal
        item = {
            "fissure_score": float(score),
            "fissure_area_ratio": float(area_ratio),
            "fissure_aspect": float(aspect),
            "fissure_horizontal": float(horizontal),
        }
        if best is None or item["fissure_score"] > best["fissure_score"]:
            best = item
    return best if best is not None else _empty_fissure()


def _empty_fissure() -> dict[str, float]:
    return {
        "fissure_score": 0.0,
        "fissure_area_ratio": 0.0,
        "fissure_aspect": 0.0,
        "fissure_horizontal": 0.0,
    }


def auto_find_eye_roi(gray: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict[str, Any]]]:
    height, width = gray.shape
    params = full_image_scale_params(gray.shape)
    enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    low_full = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=params["full_blur_sigma"])
    kernel_size = int(params["full_blackhat_kernel"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(low_full, cv2.MORPH_BLACKHAT, kernel)
    _, dark_mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_mask = cv2.morphologyEx(
        dark_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    seed_candidates: list[dict[str, Any]] = []
    fallback_candidates: list[dict[str, Any]] = []
    bright_threshold = max(220, float(np.percentile(gray, 99.3)))

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < params["seed_area_min"] or area > params["fallback_area_max"]:
            continue
        x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
        if x <= 2 or y <= 2 or x + bbox_w >= width - 2 or y + bbox_h >= height - 2:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter**2)
        if circularity < 0.15:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        dark_score = float(np.mean(blackhat[y : y + bbox_h, x : x + bbox_w]))

        if params["seed_area_min"] <= area <= params["seed_area_max"] and circularity >= 0.45:
            roi_w = max(int(params["seed_min_roi_w"]), int(bbox_w * 4.0))
            roi_h = max(int(params["seed_min_roi_h"]), int(bbox_h * 4.0))
            roi_x = max(0, min(int(cx - roi_w * 0.52), width - 1))
            roi_y = max(0, min(int(cy - roi_h * 0.52), height - 1))
            roi_w = min(roi_w, width - roi_x)
            roi_h = min(roi_h, height - roi_y)
            roi_gray = gray[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            fissure = estimate_eye_fissure_features(roi_gray)
            seed_mask = np.zeros_like(gray)
            cv2.drawContours(seed_mask, [contour], -1, 255, -1)
            mean_seed_intensity = float(cv2.mean(gray, mask=seed_mask)[0])
            bright_ratio = float(np.mean(roi_gray > bright_threshold))
            center_prior = 1 - abs(cx - width * 0.55) / (width * 0.55)
            score = (
                dark_score
                + 25 * circularity
                + 0.01 * area
                - 0.22 * mean_seed_intensity
                + 8 * min(bright_ratio, 0.03)
                + 12 * fissure["fissure_score"]
                + 4 * center_prior
            )
            seed_candidates.append(
                {
                    "score": float(score),
                    "contour": contour,
                    "bbox": (x, y, bbox_w, bbox_h),
                    "roi": (roi_x, roi_y, roi_w, roi_h),
                    "area": float(area),
                    "circularity": float(circularity),
                    "dark_score": dark_score,
                    "bright_ratio": bright_ratio,
                    "mean_seed_intensity": mean_seed_intensity,
                    "mode": "dark circular seed",
                    **fissure,
                }
            )

        if params["fallback_area_min"] <= area <= params["fallback_area_max"]:
            roi = expand_bbox(x, y, bbox_w, bbox_h, gray.shape, scale_x=2.5, scale_y=2.2)
            roi_x, roi_y, roi_w, roi_h = roi
            roi_gray = gray[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            fissure = estimate_eye_fissure_features(roi_gray)
            bright_ratio = float(np.mean(roi_gray > bright_threshold))
            score = (
                dark_score
                + 60 * min(bright_ratio, 0.02)
                + 0.002 * area
                + 15 * circularity
                + 12 * fissure["fissure_score"]
            )
            fallback_candidates.append(
                {
                    "score": float(score),
                    "contour": contour,
                    "bbox": (x, y, bbox_w, bbox_h),
                    "roi": roi,
                    "area": float(area),
                    "circularity": float(circularity),
                    "dark_score": dark_score,
                    "bright_ratio": bright_ratio,
                    "mode": "fallback dark region",
                    **fissure,
                }
            )

    candidates = seed_candidates if seed_candidates else fallback_candidates
    if not candidates:
        raise RuntimeError("No eye ROI candidate found.")
    best = max(candidates, key=lambda candidate: candidate["score"])
    return best["roi"], candidates


def compute_pupil_qc(result: dict[str, Any], roi_shape: tuple[int, int]) -> dict[str, Any]:
    height, width = roi_shape
    candidate = result["candidate"]
    contour = result["contour_roi"]
    cx_roi, cy_roi = result["center_roi"]
    radius = result["radius_px"]
    residual = circle_fit_residual(contour, cx_roi, cy_roi, radius)
    edge_margin = min(cx_roi, cy_roi, width - cx_roi, height - cy_roi)
    score_values = sorted(c["score"] for c in result["candidates"])
    score_margin = score_values[1] - score_values[0] if len(score_values) > 1 else np.nan
    area = candidate["area"]
    circle_area = np.pi * radius * radius
    area_ratio_to_circle = area / circle_area if circle_area > 0 else np.nan

    warnings: list[str] = []
    if candidate["circularity"] < 0.5:
        warnings.append("low_circularity")
    if residual > max(1.5, 0.18 * radius):
        warnings.append("high_circle_residual")
    if edge_margin < max(3, 0.25 * radius):
        warnings.append("pupil_near_roi_edge")
    if not np.isnan(score_margin) and score_margin < 3:
        warnings.append("ambiguous_candidate_score")
    if area_ratio_to_circle < 0.35 or area_ratio_to_circle > 1.35:
        warnings.append("contour_circle_area_mismatch")

    residual_ref = max(1.5, 0.18 * radius)
    circularity_score = float(np.clip((candidate["circularity"] - 0.35) / 0.55, 0, 1))
    residual_score = float(np.clip(1 - residual / (2 * residual_ref), 0, 1))
    edge_score = float(np.clip(edge_margin / max(3, 1.5 * radius), 0, 1))
    margin_score = 0.65 if np.isnan(score_margin) else float(np.clip(score_margin / 10, 0, 1))
    area_score = float(np.clip(1 - abs(area_ratio_to_circle - 1) / 0.65, 0, 1))
    confidence_raw = (
        0.25 * circularity_score
        + 0.25 * residual_score
        + 0.20 * edge_score
        + 0.15 * margin_score
        + 0.15 * area_score
    )
    confidence = float(np.clip(confidence_raw - 0.12 * len(warnings), 0, 1))
    confidence_level = "high" if confidence >= 0.85 else "medium" if confidence >= 0.65 else "low"

    return {
        "qc_pass": len(warnings) == 0,
        "qc_confidence": confidence,
        "qc_confidence_level": confidence_level,
        "qc_warnings": ";".join(warnings),
        "qc_circularity": float(candidate["circularity"]),
        "qc_contour_area": float(area),
        "qc_circle_residual_px": residual,
        "qc_edge_margin_px": float(edge_margin),
        "qc_score_margin": float(score_margin) if not np.isnan(score_margin) else np.nan,
        "qc_area_ratio_to_circle": float(area_ratio_to_circle),
        "qc_num_candidates": len(result["candidates"]),
        "qc_circularity_score": circularity_score,
        "qc_residual_score": residual_score,
        "qc_edge_score": edge_score,
        "qc_margin_score": margin_score,
        "qc_area_score": area_score,
    }


def extract_pupil_from_roi(gray: np.ndarray, roi: tuple[int, int, int, int]) -> dict[str, Any]:
    x, y, width, height = roi
    roi_raw = gray[y : y + height, x : x + width].copy()
    params = roi_scale_params(roi_raw.shape)
    roi_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(roi_raw)
    glint_threshold = max(220, float(np.percentile(roi_raw, 99.3)))
    params["glint_threshold"] = glint_threshold
    glint_mask = (roi_raw > glint_threshold).astype(np.uint8) * 255
    glint_mask = cv2.dilate(
        glint_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    roi_no_glint = cv2.inpaint(
        roi_enhanced, glint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )
    low_roi = cv2.GaussianBlur(roi_no_glint, (0, 0), sigmaX=params["roi_blur_sigma"])
    kernel_size = int(params["pupil_blackhat_kernel"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(low_roi, cv2.MORPH_BLACKHAT, kernel)
    _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    roi_center = np.array([width * 0.45, height * 0.45])
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < params["pupil_area_min"] or area > params["pupil_area_max"]:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter**2)
        if circularity < 0.35:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        mask = np.zeros_like(roi_raw)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_intensity = cv2.mean(low_roi, mask=mask)[0]
        dist_to_center = np.linalg.norm(np.array([cx, cy]) - roi_center)
        score = mean_intensity - 35 * circularity + 0.25 * dist_to_center + 0.01 * area
        candidates.append(
            {
                "score": float(score),
                "contour": contour,
                "area": float(area),
                "circularity": float(circularity),
                "center": (float(cx), float(cy)),
                "mean_intensity": float(mean_intensity),
            }
        )

    if not candidates:
        raise RuntimeError("No pupil contour candidate found.")
    best = min(candidates, key=lambda candidate: candidate["score"])
    pupil_cnt_roi = best["contour"]
    cx_roi, cy_roi, radius = fit_circle_least_squares(pupil_cnt_roi)
    output = {
        "roi": roi,
        "center_roi": (cx_roi, cy_roi),
        "center_global": (cx_roi + x, cy_roi + y),
        "radius_px": radius,
        "diameter_px": 2 * radius,
        "contour_roi": pupil_cnt_roi,
        "binary_mask": binary,
        "low_roi": low_roi,
        "candidate": best,
        "candidates": candidates,
        "scale_params": params,
    }
    output["qc"] = compute_pupil_qc(output, roi_raw.shape)
    output["result_image"] = annotate_result(gray, output)
    return output


def analyze_image(gray_or_bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    gray = ensure_gray(gray_or_bgr)
    if roi is None:
        roi, roi_candidates = auto_find_eye_roi(gray)
    else:
        roi_candidates = []
    result = extract_pupil_from_roi(gray, roi)
    result["roi_candidates"] = roi_candidates
    return result


def ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def annotate_result(gray_or_bgr: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    gray = ensure_gray(gray_or_bgr)
    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x, y, width, height = result["roi"]
    cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 0, 255), 2)
    contour_global = result["contour_roi"] + np.array([[[x, y]]])
    cv2.drawContours(annotated, [contour_global], -1, (0, 255, 0), 2)
    cx, cy = result["center_global"]
    radius = result["radius_px"]
    cv2.circle(annotated, (int(round(cx)), int(round(cy))), int(round(radius)), (255, 0, 0), 2)
    cv2.circle(annotated, (int(round(cx)), int(round(cy))), 2, (0, 255, 255), -1)
    qc = result["qc"]
    label = f"diam={result['diameter_px']:.1f}px  QC={qc['qc_confidence']:.2f} ({qc['qc_confidence_level']})"
    cv2.putText(
        annotated,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def result_to_row(
    result: dict[str, Any],
    frame_index: int | None = None,
    time_s: float | None = None,
    micron_per_px: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "success": True,
        "frame_index": frame_index,
        "time_s": time_s,
        "roi_x": result["roi"][0],
        "roi_y": result["roi"][1],
        "roi_w": result["roi"][2],
        "roi_h": result["roi"][3],
        "center_x_px": result["center_global"][0],
        "center_y_px": result["center_global"][1],
        "radius_px": result["radius_px"],
        "diameter_px": result["diameter_px"],
        **result["qc"],
    }
    if micron_per_px is not None:
        row["diameter_um"] = result["diameter_px"] * micron_per_px
    return row


def failed_row(frame_index: int, time_s: float | None, error: Exception) -> dict[str, Any]:
    return {
        "success": False,
        "frame_index": frame_index,
        "time_s": time_s,
        "error": f"{type(error).__name__}: {error}",
    }


def save_image_outputs(
    image_path: str | Path,
    output_dir: str | Path,
    micron_per_px: float | None = None,
) -> pd.DataFrame:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    result = analyze_image(gray)
    row = result_to_row(result, micron_per_px=micron_per_px)
    row["image"] = str(image_path)
    summary = pd.DataFrame([row])
    cv2.imwrite(str(output_dir / f"{image_path.stem}_pupil_result.png"), result["result_image"])
    summary.to_csv(output_dir / f"{image_path.stem}_pupil_summary.csv", index=False)
    return summary
