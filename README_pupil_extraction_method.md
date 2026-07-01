# Mouse Pupil Extraction and Diameter Measurement Method

## 中文版

### 方法概述

本流程用于从单张小鼠眼部或头部灰度图像中自动定位眼睛区域，提取 pupil 边界，并计算 pupil 直径。图像处理和测量使用 Python、OpenCV、NumPy、pandas 和 Matplotlib 完成。算法采用两阶段策略：第一阶段在整张图像中自动寻找包含 pupil 的眼睛 ROI；第二阶段在 ROI 内去除角膜反光、增强暗圆形结构、提取 pupil 轮廓，并用最小二乘法拟合圆形以得到 pupil 半径和直径。

### 输入与输出

输入为一张 8-bit 灰度图像，或可由 OpenCV 读取并转换为灰度的图像文件。当前 notebook 默认分析 `pupiltest.jpeg`。输出包括自动检测到的 eye ROI、pupil 中心坐标、pupil 半径、pupil 直径、最终叠加可视化图，以及包含测量结果的 CSV 文件。

输出字段包括：

- `roi_x`, `roi_y`, `roi_w`, `roi_h`: 自动检测到的眼睛 ROI，格式为左上角坐标和宽高。
- `center_x_px`, `center_y_px`: pupil 圆心在原始图像中的像素坐标。
- `radius_px`: 拟合得到的 pupil 半径，单位为像素。
- `diameter_px`: 拟合得到的 pupil 直径，单位为像素。
- `diameter_um`: 如果提供 `micron_per_px`，可进一步换算为微米。
- `qc_pass`, `qc_confidence`, `qc_confidence_level` 和 `qc_*`: 自动质量控制指标，包括置信度、圆度、轮廓面积、圆拟合残差、ROI 边缘距离、候选分数间隔和候选数量。

此外，流程会输出 ROI 候选 cluster 图。该图不是训练模型，而是将 ROI 候选的数值特征标准化后，用 SVD/PCA 投影到二维空间，再进行轻量 KMeans-style 聚类。最终被选择的 ROI 以红色星号标出，用于检查自动选择是否与其它背景候选明显分离。

ROI cluster 图使用的 features 为：

- `score`: ROI 候选综合评分。
- `area`: 暗结构候选轮廓面积。
- `circularity`: 暗结构候选圆度。
- `dark_score`: 候选 bbox 内 black-hat 响应均值。
- `bright_ratio`: 扩展 ROI 内强高亮像素比例。
- `mean_seed_intensity`: 暗圆种子区域在原始灰度图中的平均强度，仅种子候选有该值；缺失值在 cluster 可视化中填 0。
- `fissure_score`: 横向暗眼裂综合评分。
- `fissure_area_ratio`: ROI 内暗眼裂候选区域面积占 ROI 面积的比例。
- `fissure_aspect`: 暗眼裂候选 bbox 的宽高比。
- `fissure_horizontal`: 横向延展性评分。

PCA 不是直接把某一个 feature 画成横轴或纵轴，而是先对上述 features 做 z-score 标准化，然后用 SVD 求主成分。PC1 和 PC2 是这些标准化 features 的线性组合：

```text
PC1 = sum(z_feature_i * PC1_loading_i)
PC2 = sum(z_feature_i * PC2_loading_i)
```

notebook 会输出 `PC loadings` 表。某个 feature 在 PC1 或 PC2 上的 loading 绝对值越大，说明它对该 PC 轴的贡献越大；loading 的正负号表示该 feature 与该 PC 方向的正向或反向关系。notebook 也会输出 `PC explained variance ratio`，表示 PC1 和 PC2 分别解释候选特征变异的比例。

每次生成 cluster 图时，图下方的外部说明区域会自动标注当前数据的 PC 组成公式，避免遮挡候选点。为避免文字过长，图中使用短变量名：`circ` = `circularity`，`dark` = `dark_score`，`bright` = `bright_ratio`，`seed_int` = `mean_seed_intensity`，`fis_score` = `fissure_score`，`fis_area` = `fissure_area_ratio`，`fis_aspect` = `fissure_aspect`，`fis_horiz` = `fissure_horizontal`。

### 第一阶段：自动定位眼睛 ROI

为了避免背景中的大面积强反光结构被误识别为眼睛，本流程优先寻找“小而圆的暗种子点”作为 pupil 或 iris 候选，然后围绕该候选点扩展得到眼睛 ROI。若未找到合适的暗圆种子点，则回退到较大暗区域搜索策略。

首先，对整张灰度图像进行 CLAHE 自适应直方图均衡化，以增强局部对比度。随后使用高斯低通滤波降低毛发、噪声和细小纹理的影响。接着使用 black-hat 形态学变换增强局部暗结构。black-hat 变换定义为形态学闭运算结果减去原始图像，因此对比周围更暗的小区域会得到较强响应。

black-hat 响应图使用 Otsu 方法自动阈值化，得到候选暗结构二值 mask。对 mask 执行形态学开运算以去除小噪声，然后提取外部轮廓。每个轮廓根据面积、位置和圆度进行筛选。面积阈值和 kernel 尺寸不是固定像素值，而是按图像面积和短边长度自适应缩放。圆度定义为：

```text
circularity = 4 * pi * area / perimeter^2
```

优先的暗圆种子候选满足尺度自适应面积范围：

```text
seed_area_min = max(20, image_area * 0.00012)
seed_area_max = max(80, image_area * 0.0028)
seed_area_min <= area <= seed_area_max
circularity >= 0.45
```

整图 black-hat kernel 同样按短边缩放：

```text
full_blackhat_kernel = odd(max(31, short_side * 0.082))
full_blur_sigma = max(2.0, short_side * 0.0064)
```

贴近图像边界的候选会被排除，以减少边缘伪影。对每个候选，计算其轮廓中心、black-hat 响应强度、原始灰度平均强度、候选 ROI 内高亮像素比例、弱中心先验，以及新增的眼裂特征。眼裂特征基于候选 ROI 内暗区域的横向延展性：ROI 内较大的暗区域若呈横向椭圆或横向长条形，则 `fissure_score` 较高。候选评分为：

```text
score =
    dark_score
    + 25 * circularity
    + 0.01 * area
    - 0.22 * mean_seed_intensity
    + 8 * min(bright_ratio, 0.03)
    + 12 * fissure_score
    + 4 * center_prior
```

其中 `dark_score` 是候选 bbox 内 black-hat 响应的平均值，`mean_seed_intensity` 是候选区域在原始灰度图中的平均强度，`bright_ratio` 是扩展 ROI 内强高亮像素比例，`fissure_score` 描述 ROI 内是否存在横向暗椭圆眼裂，`center_prior` 是用于打破毛发或背景候选平局的弱先验。该评分偏好黑帽响应强、形状接近圆形、原始强度较暗、面积适中、周围存在横向眼裂结构，并允许 ROI 内存在少量角膜反光。

最佳暗圆候选确定后，根据候选中心和候选 bbox 尺寸扩展 ROI：

```text
roi_w = max(70, 4.0 * candidate_width)
roi_h = max(60, 4.0 * candidate_height)
roi_x = candidate_center_x - 0.52 * roi_w
roi_y = candidate_center_y - 0.52 * roi_h
```

ROI 会被限制在原始图像边界内。对于示例图像 `pupiltest.jpeg`，自动检测得到的 ROI 为 `(375, 263, 76, 80)`。

如果没有任何暗圆种子候选，算法会使用回退策略：选择面积在 300 到 6000 像素之间、圆度大于 0.15 的较大暗区域候选，并通过 `expand_bbox` 将其扩展为眼睛 ROI。回退候选评分基于 black-hat 响应、少量高亮反光、候选面积和圆度。

### 第二阶段：ROI 内 pupil 边界提取

在自动定位的眼睛 ROI 内，首先提取原始灰度 ROI。为了增强 ROI 中的局部对比度，对 ROI 再次使用 CLAHE。角膜反光点会干扰 pupil 边界提取，因此先根据 ROI 内强度分布自适应生成高亮反光 mask。当前阈值为 `max(220, ROI 第 99.3 百分位灰度值)`，并使用 5 x 5 椭圆结构元素膨胀一轮。随后使用 Telea inpainting 方法对反光区域进行修补。

去除反光后的 ROI 使用尺度自适应高斯滤波进一步平滑，`roi_blur_sigma = max(1.0, roi_short_side * 0.025)`。之后使用按 ROI 短边缩放的椭圆结构元素进行 black-hat 形态学变换，`pupil_blackhat_kernel = odd(max(15, roi_short_side * 0.32))`，以增强 ROI 内的暗圆形 pupil 结构。black-hat 响应图经 Otsu 阈值化后得到 pupil 候选二值 mask。该 mask 依次经过 3 x 3 椭圆结构元素的开运算和 5 x 5 椭圆结构元素的闭运算，以去除噪声并连接边界。

从二值 mask 中提取候选轮廓。候选轮廓需满足尺度自适应面积范围：

```text
pupil_area_min = max(8, roi_area * 0.003)
pupil_area_max = max(80, roi_area * 0.14)
pupil_area_min <= area <= pupil_area_max
circularity >= 0.35
```

对每个候选轮廓，计算轮廓面积、圆度、中心位置、在平滑 ROI 中的平均灰度，以及其中心到 ROI 参考中心 `(0.45 * roi_w, 0.45 * roi_h)` 的距离。候选评分为：

```text
score =
    mean_intensity
    - 35 * circularity
    + 0.25 * dist_to_center
    + 0.01 * area
```

该评分越小越好，偏好更暗、更圆、靠近 ROI 中心区域且面积适中的轮廓。得分最低的轮廓被选为 pupil 边界。

### 圆拟合与直径计算

选中的 pupil 轮廓点使用最小二乘法拟合圆。圆模型为：

```text
(x - cx)^2 + (y - cy)^2 = r^2
```

将其展开为线性形式：

```text
2 * cx * x + 2 * cy * y + d = x^2 + y^2
```

通过最小二乘求解 `cx`、`cy` 和 `d`，再计算半径：

```text
r = sqrt(cx^2 + cy^2 + d)
```

pupil 直径为：

```text
diameter_px = 2 * r
```

圆心坐标会从 ROI 坐标系转换回原始图像坐标系。若已知空间校准比例 `micron_per_px`，则可进一步计算：

```text
diameter_um = diameter_px * micron_per_px
```

### 示例结果

对 `pupiltest.jpeg` 的当前自动分析结果为：

```text
ROI = (375, 263, 76, 80)
pupil center = (415.844, 303.445) px
radius = 8.907 px
diameter = 17.815 px
QC confidence = 0.949, high
```

结果文件保存在 `pupil_output/` 中：

- `pupiltest_pupil_result.png`: 原图上叠加 eye ROI、pupil 轮廓、拟合圆和圆心。
- `pupiltest_pupil_summary.csv`: pupil 测量结果表。
- `pupiltest_roi_candidate_clusters.png`: ROI 候选特征 cluster 图，红色星号为最终选中的 ROI。

### 自动质量控制、cluster 图与手动校正

运行 notebook 时建议保留 `debug = True`，逐步检查 CLAHE 图、black-hat 响应图、二值 mask、自动 ROI、ROI 内候选 mask、候选 cluster 图以及最终叠加图。自动 QC 会输出：

- `qc_confidence`: 0 到 1 的启发式置信度分数，越高表示识别结果越可信。
- `qc_confidence_level`: 根据置信度分为 `high`, `medium`, `low`。
- `qc_circularity`: 被选 pupil 轮廓的圆度。
- `qc_contour_area`: 被选 pupil 轮廓面积。
- `qc_circle_residual_px`: 圆拟合平均绝对径向残差。
- `qc_edge_margin_px`: 拟合圆心到 ROI 边缘的最小距离。
- `qc_score_margin`: 最优 pupil 候选与第二候选的分数间隔。
- `qc_area_ratio_to_circle`: 轮廓面积与拟合圆面积的比例。
- `qc_num_candidates`: ROI 内通过筛选的 pupil 候选数量。
- `qc_pass` 和 `qc_warnings`: 根据上述指标生成的自动质控结论和警告标签。

`qc_confidence` 是可解释的 QC 综合分数，不是由人工标注数据校准出的统计概率。它由五个归一化子分数组成：

```text
qc_confidence =
    0.25 * circularity_score
    + 0.25 * residual_score
    + 0.20 * edge_score
    + 0.15 * margin_score
    + 0.15 * area_score
    - 0.12 * number_of_qc_warnings
```

其中 `circularity_score` 奖励更圆的轮廓，`residual_score` 奖励更小的圆拟合残差，`edge_score` 奖励 pupil 远离 ROI 边界，`margin_score` 奖励最优候选明显优于第二候选，`area_score` 奖励轮廓面积与拟合圆面积一致。置信度被限制在 `[0, 1]`。等级定义为：

```text
high:   qc_confidence >= 0.85
medium: 0.65 <= qc_confidence < 0.85
low:    qc_confidence < 0.65
```

最终叠加图左上角会标注 pupil 直径、`QC` 置信度和等级。

若自动 ROI 失败，可以在 notebook 最后一节手动指定 ROI：

```python
manual_roi = (x, y, w, h)
result_manual = extract_pupil_from_roi(gray, manual_roi, debug=True)
```

在图像亮度、视角、pupil 大小或分辨率明显变化时，优先检查 cluster 图和 QC 警告，再调整面积比例、black-hat kernel 比例、反光百分位阈值或 ROI 扩展倍数。

---

## English Version

### Method Overview

This workflow was designed to automatically localize the eye region, extract the pupil boundary, and estimate pupil diameter from a single grayscale image of a mouse eye or head. Image processing and measurement were implemented in Python using OpenCV, NumPy, pandas, and Matplotlib. The pipeline consists of two stages. First, an eye region of interest (ROI) containing the pupil is automatically detected from the full image. Second, within the ROI, corneal glints are removed, dark circular structures are enhanced, the pupil contour is segmented, and a circle is fitted to the selected contour by least squares to estimate pupil radius and diameter.

### Input and Output

The input is an 8-bit grayscale image, or any image file that can be read and converted to grayscale by OpenCV. The current notebook analyzes `pupiltest.jpeg` by default. The outputs include the detected eye ROI, pupil center coordinates, pupil radius, pupil diameter, an overlay image for visual inspection, and a CSV file containing the measurement results.

The output fields are:

- `roi_x`, `roi_y`, `roi_w`, `roi_h`: detected eye ROI, represented by the upper-left coordinate and width/height.
- `center_x_px`, `center_y_px`: pupil center coordinates in the original image.
- `radius_px`: fitted pupil radius in pixels.
- `diameter_px`: fitted pupil diameter in pixels.
- `diameter_um`: optional physical diameter if `micron_per_px` is provided.
- `qc_pass`, `qc_confidence`, `qc_confidence_level`, and `qc_*`: automatic quality-control metrics, including confidence, contour circularity, contour area, circle-fit residual, ROI edge margin, candidate score margin, and candidate count.

The workflow also exports an ROI candidate cluster plot. This plot is not a trained model. It standardizes candidate-level numerical features, projects them into two dimensions using SVD/PCA, and applies lightweight KMeans-style clustering. The final selected ROI is marked with a red star, allowing visual inspection of whether the selected eye candidate is separated from background candidates.

The ROI cluster plot uses the following features:

- `score`: composite ROI candidate score.
- `area`: contour area of the dark candidate structure.
- `circularity`: circularity of the dark candidate structure.
- `dark_score`: mean black-hat response within the candidate bounding box.
- `bright_ratio`: fraction of very bright pixels within the expanded ROI.
- `mean_seed_intensity`: mean raw grayscale intensity of the dark seed region; this value exists for seed candidates, and missing values are filled with 0 for cluster visualization.
- `fissure_score`: composite score for a horizontal dark eye fissure.
- `fissure_area_ratio`: area fraction of the dark fissure candidate within the ROI.
- `fissure_aspect`: width-to-height ratio of the dark fissure candidate bounding box.
- `fissure_horizontal`: horizontal elongation score.

PCA does not use one raw feature as the x- or y-axis. Instead, the listed features are first z-score standardized, and SVD is then used to calculate principal components. PC1 and PC2 are linear combinations of the standardized features:

```text
PC1 = sum(z_feature_i * PC1_loading_i)
PC2 = sum(z_feature_i * PC2_loading_i)
```

The notebook reports a `PC loadings` table. A feature with a larger absolute loading contributes more strongly to that PC axis; the sign of the loading indicates whether the feature varies in the positive or negative direction of that PC. The notebook also reports the `PC explained variance ratio`, which indicates how much candidate-feature variance is explained by PC1 and PC2.

Each time the cluster plot is generated, the current PC composition formulas are automatically annotated in an external note area below the plot to avoid covering candidate points. To keep the plot readable, shortened variable names are used in the annotation: `circ` = `circularity`, `dark` = `dark_score`, `bright` = `bright_ratio`, `seed_int` = `mean_seed_intensity`, `fis_score` = `fissure_score`, `fis_area` = `fissure_area_ratio`, `fis_aspect` = `fissure_aspect`, and `fis_horiz` = `fissure_horizontal`.

### Stage 1: Automatic Eye ROI Detection

To avoid selecting large bright background structures as the eye, the current implementation first searches for a small, round, dark seed corresponding to the pupil or iris, and then expands an ROI around that seed. If no suitable dark circular seed is found, the algorithm falls back to a larger dark-region ROI search.

The full grayscale image is first processed with contrast-limited adaptive histogram equalization (CLAHE) to enhance local contrast. Gaussian low-pass filtering is then applied to reduce hair-like texture, fine noise, and small high-frequency structures. A black-hat morphological transform is applied to enhance local dark structures. The black-hat transform is defined as the difference between the morphological closing of the image and the image itself; therefore, small regions that are darker than their local surroundings show a strong response.

The black-hat response image is thresholded using Otsu's method to generate a binary mask of candidate dark structures. Morphological opening is applied to remove small noise, and external contours are extracted. Each contour is screened by area, border position, and circularity. Area thresholds and kernel sizes are scaled adaptively from image size rather than kept as fixed pixel values. Circularity is defined as:

```text
circularity = 4 * pi * area / perimeter^2
```

Primary dark circular seed candidates must satisfy a scale-adaptive area range:

```text
seed_area_min = max(20, image_area * 0.00012)
seed_area_max = max(80, image_area * 0.0028)
seed_area_min <= area <= seed_area_max
circularity >= 0.45
```

The full-image black-hat kernel is also scaled from the image short side:

```text
full_blackhat_kernel = odd(max(31, short_side * 0.082))
full_blur_sigma = max(2.0, short_side * 0.0064)
```

Candidates touching the image border are excluded to reduce edge artifacts. For each candidate, the contour center, black-hat response, mean raw intensity, bright-pixel fraction in the expanded ROI, a weak center prior, and the horizontal eye-fissure feature are computed. The fissure feature measures whether the ROI contains a horizontally elongated dark elliptical region, as expected from the mouse palpebral fissure. The candidate score is:

```text
score =
    dark_score
    + 25 * circularity
    + 0.01 * area
    - 0.22 * mean_seed_intensity
    + 8 * min(bright_ratio, 0.03)
    + 12 * fissure_score
    + 4 * center_prior
```

Here, `dark_score` is the mean black-hat response within the candidate bounding box, `mean_seed_intensity` is the mean raw grayscale intensity of the candidate region, `bright_ratio` is the fraction of very bright pixels in the expanded ROI, `fissure_score` describes the presence of a horizontal dark eye fissure, and `center_prior` is a weak prior used only to break ties between hair or background candidates. This score favors strong black-hat responses, high circularity, low raw intensity, moderate area, the presence of a horizontal fissure-like dark structure, and allows a small amount of corneal glint within the ROI.

After the best seed is selected, the eye ROI is expanded around the candidate center:

```text
roi_w = max(70, 4.0 * candidate_width)
roi_h = max(60, 4.0 * candidate_height)
roi_x = candidate_center_x - 0.52 * roi_w
roi_y = candidate_center_y - 0.52 * roi_h
```

The ROI is clipped to remain inside the original image. For the example image `pupiltest.jpeg`, the automatically detected ROI was `(375, 263, 76, 80)`.

If no primary dark circular seed is detected, the fallback strategy selects larger dark-region candidates with area between 300 and 6000 pixels and circularity above 0.15, then expands the bounding box using `expand_bbox`. The fallback score is based on black-hat response, the presence of a small amount of bright glint, contour area, and circularity.

### Stage 2: Pupil Boundary Extraction Within the ROI

The raw grayscale ROI is extracted from the original image. CLAHE is then applied within the ROI to enhance local contrast. Because corneal glints may interfere with pupil boundary detection, a glint mask is generated adaptively from the ROI intensity distribution. The current threshold is `max(220, ROI 99.3rd percentile intensity)`. This mask is dilated once using a 5 x 5 elliptical structuring element, and the masked region is repaired using Telea inpainting.

The glint-corrected ROI is smoothed with scale-adaptive Gaussian filtering, `roi_blur_sigma = max(1.0, roi_short_side * 0.025)`. A black-hat morphological transform with an ROI-scaled elliptical kernel, `pupil_blackhat_kernel = odd(max(15, roi_short_side * 0.32))`, is then applied to enhance the dark circular pupil structure. The black-hat response is thresholded using Otsu's method to obtain a binary pupil candidate mask. The mask is processed by morphological opening with a 3 x 3 elliptical kernel and morphological closing with a 5 x 5 elliptical kernel to remove noise and connect fragmented boundaries.

Contours are extracted from the binary mask. Candidate pupil contours must satisfy a scale-adaptive area range:

```text
pupil_area_min = max(8, roi_area * 0.003)
pupil_area_max = max(80, roi_area * 0.14)
pupil_area_min <= area <= pupil_area_max
circularity >= 0.35
```

For each candidate, contour area, circularity, centroid, mean intensity in the smoothed ROI, and distance to the ROI reference center `(0.45 * roi_w, 0.45 * roi_h)` are computed. The pupil candidate score is:

```text
score =
    mean_intensity
    - 35 * circularity
    + 0.25 * dist_to_center
    + 0.01 * area
```

Lower scores are better. This scoring function favors contours that are darker, more circular, closer to the expected central eye region, and of moderate area. The lowest-scoring contour is selected as the pupil boundary.

### Circle Fitting and Diameter Estimation

The selected pupil contour points are fitted with a circle using least squares. The circle model is:

```text
(x - cx)^2 + (y - cy)^2 = r^2
```

This is rearranged into a linear least-squares problem:

```text
2 * cx * x + 2 * cy * y + d = x^2 + y^2
```

After solving for `cx`, `cy`, and `d`, the radius is calculated as:

```text
r = sqrt(cx^2 + cy^2 + d)
```

The pupil diameter is:

```text
diameter_px = 2 * r
```

The fitted center is converted from ROI coordinates back to original image coordinates. If a spatial calibration factor `micron_per_px` is available, the physical diameter can be calculated as:

```text
diameter_um = diameter_px * micron_per_px
```

### Example Result

For `pupiltest.jpeg`, the current automatic analysis produced:

```text
ROI = (375, 263, 76, 80)
pupil center = (415.844, 303.445) px
radius = 8.907 px
diameter = 17.815 px
QC confidence = 0.949, high
```

Output files are saved in `pupil_output/`:

- `pupiltest_pupil_result.png`: original image overlaid with the eye ROI, pupil contour, fitted circle, and fitted center.
- `pupiltest_pupil_summary.csv`: table of pupil measurement results.
- `pupiltest_roi_candidate_clusters.png`: ROI candidate feature cluster plot, with the final selected ROI marked by a red star.

### Automatic Quality Control, Cluster Plot, and Manual Correction

For quality control, `debug = True` should be used to inspect the CLAHE image, black-hat response, binary masks, automatically detected ROI, ROI-level candidate mask, candidate cluster plot, and final overlay image. Automatic QC includes:

- `qc_confidence`: heuristic confidence score from 0 to 1; larger values indicate a more reliable detection.
- `qc_confidence_level`: confidence category, reported as `high`, `medium`, or `low`.
- `qc_circularity`: circularity of the selected pupil contour.
- `qc_contour_area`: area of the selected pupil contour.
- `qc_circle_residual_px`: mean absolute radial residual of the circle fit.
- `qc_edge_margin_px`: minimum distance from the fitted circle center to the ROI boundary.
- `qc_score_margin`: score difference between the best and second-best pupil candidates.
- `qc_area_ratio_to_circle`: ratio between contour area and fitted circle area.
- `qc_num_candidates`: number of pupil candidates passing the filters within the ROI.
- `qc_pass` and `qc_warnings`: automatic QC decision and warning labels generated from these metrics.

`qc_confidence` is an interpretable QC summary score, not a statistically calibrated probability derived from manual labels. It combines five normalized component scores:

```text
qc_confidence =
    0.25 * circularity_score
    + 0.25 * residual_score
    + 0.20 * edge_score
    + 0.15 * margin_score
    + 0.15 * area_score
    - 0.12 * number_of_qc_warnings
```

Here, `circularity_score` rewards a more circular contour, `residual_score` rewards a lower circle-fit residual, `edge_score` rewards a pupil center farther from the ROI boundary, `margin_score` rewards a clearer separation between the best and second-best candidates, and `area_score` rewards agreement between the contour area and fitted-circle area. The confidence score is clipped to `[0, 1]`. Confidence levels are defined as:

```text
high:   qc_confidence >= 0.85
medium: 0.65 <= qc_confidence < 0.85
low:    qc_confidence < 0.65
```

The final overlay image annotates the pupil diameter, `QC` confidence score, and confidence level in the upper-left corner.

If automatic ROI detection fails, a manual ROI can be provided in the final notebook section:

```python
manual_roi = (x, y, w, h)
result_manual = extract_pupil_from_roi(gray, manual_roi, debug=True)
```

If image brightness, viewing angle, pupil size, or image resolution differs substantially, the candidate cluster plot and QC warnings should be inspected first. The area ratios, black-hat kernel scaling factors, glint percentile threshold, or ROI expansion factors can then be adjusted if needed.
