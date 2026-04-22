# import cv2, json
# import numpy as np
# from pycocotools import mask as mask_utils

# # ==== 配置 ====
# json_path = "/home/avent/Desktop/generated_data/2026-04-22-111856/coco_annotations_yblarjvw.json"
# image_root = "/home/avent/Desktop/generated_data/2026-04-22-111856/"   # JSON 里 file_name 的根目录

# # ==== 读取 ====
# with open(json_path, "r") as f:
#     coco = json.load(f)

# images = {img["id"]: img for img in coco["images"]}

# # 按 image_id 分组
# ann_by_image = {}
# for ann in coco["annotations"]:
#     ann_by_image.setdefault(ann["image_id"], []).append(ann)

# # ==== 可视化 ====
# for image_id, anns in ann_by_image.items():
#     img_info = images[image_id]
#     img_path = image_root + img_info["file_name"]

#     img = cv2.imread(img_path)
#     if img is None:
#         print("fail:", img_path)
#         continue

#     overlay = img.copy()

#     for ann in anns:
#         # ---- decode mask ----
#         seg = ann["segmentation"]

#         if isinstance(seg, dict):  # RLE
#             rle = seg.copy()
#             if isinstance(rle["counts"], str):
#                 rle["counts"] = rle["counts"].encode("utf-8")

#             mask = mask_utils.decode(rle)
#         else:
#             continue  # polygon暂不处理

#         # ---- 随机颜色 ----
#         color = np.random.randint(0, 255, 3).tolist()

#         # ---- 叠加 mask ----
#         overlay[mask == 1] = overlay[mask == 1] * 0.5 + np.array(color) * 0.5

#         # ---- bbox ----
#         x, y, w, h = map(int, ann["bbox"])
#         cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

#     # ---- 显示 ----
#     vis = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
#     cv2.imshow("vis", vis)

#     key = cv2.waitKey(0)
#     if key == 27:  # ESC退出
#         break

# cv2.destroyAllWindows()


import json
import cv2
import numpy as np
from pycocotools import mask as mask_utils

# ==== 配置 ====
json_path = "/home/avent/Desktop/generated_data/2026-04-22-145145/coco_annotations_xhtcmyyt.json"
image_root = "/home/avent/Desktop/generated_data/2026-04-22-145145/"   # JSON 里 file_name 的根目录


# ===== 读取 =====
with open(json_path, "r") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

ann_by_image = {}
for ann in coco["annotations"]:
    ann_by_image.setdefault(ann["image_id"], []).append(ann)

image_ids = list(ann_by_image.keys())
image_idx = 0

# ===== 控制参数 =====
target_names = None   # None = 显示全部，例如 {"car", "person"}
show_only_one = False
selected_instance_idx = 0


def decode_mask(seg):
    rle = seg.copy()
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return mask_utils.decode(rle)


def draw_contour(img, mask, color):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(img, contours, -1, color, 1)


while True:
    image_id = image_ids[image_idx]
    img_info = images[image_id]
    img_path = image_root + img_info["file_name"]

    img = cv2.imread(img_path)
    if img is None:
        print("fail:", img_path)
        image_idx = (image_idx + 1) % len(image_ids)
        continue

    overlay = img.copy()
    anns = ann_by_image[image_id]

    valid_anns = []
    for ann in anns:
        name = cat_id_to_name.get(ann["category_id"], "unknown")
        if target_names is None or name in target_names:
            valid_anns.append(ann)

    if show_only_one and valid_anns:
        valid_anns = [valid_anns[selected_instance_idx % len(valid_anns)]]

    for i, ann in enumerate(valid_anns):
        seg = ann["segmentation"]
        if not isinstance(seg, dict):
            continue

        mask = decode_mask(seg)

        color = np.random.randint(0, 255, 3).tolist()

        # ===== mask 半透明填充 =====
        overlay[mask == 1] = overlay[mask == 1] * 0.5 + np.array(color) * 0.5

        # ===== 轮廓（重点）=====
        draw_contour(overlay, mask, (255, 255, 255))

        # ===== bbox =====
        x, y, w, h = map(int, ann["bbox"])
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        # ===== label =====
        label = cat_id_to_name.get(ann["category_id"], "unknown")
        cv2.putText(
            overlay,
            f"{label}:{i}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    vis = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    # ===== HUD 信息 =====
    cv2.putText(
        vis,
        f"[{image_idx+1}/{len(image_ids)}] instances: {len(valid_anns)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.imshow("COCO Instance Debugger", vis)

    key = cv2.waitKey(0)

    if key == 27:  # ESC
        break
    elif key == ord("d"):  # 下一张
        image_idx = (image_idx + 1) % len(image_ids)
    elif key == ord("a"):  # 上一张
        image_idx = (image_idx - 1) % len(image_ids)
    elif key == ord("s"):  # 单实例模式
        show_only_one = not show_only_one
    elif key == ord("w"):  # 切换实例
        selected_instance_idx += 1
    elif key == ord("c"):  # 清除类别过滤
        target_names = None
    elif key == ord("1"):
        target_names = {"person"}
    elif key == ord("2"):
        target_names = {"car"}
    elif key == ord("3"):
        target_names = {"truck"}

cv2.destroyAllWindows()