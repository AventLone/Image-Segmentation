import cv2
import numpy as np

mask = cv2.imread("/media/avent/DATA/generated_data/train/2026.07.27-15:08/mask/0002.png", cv2.IMREAD_UNCHANGED)
rgb = cv2.imread("/media/avent/DATA/generated_data/train/2026.07.27-15:08/rgb/0002.png", cv2.IMREAD_COLOR)

# colorise mask
if mask is None or rgb is None:
	raise FileNotFoundError("Could not load mask or rgb image.")

if mask.ndim == 3:
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

unique_values = np.unique(mask)
rng = np.random.default_rng(42)
colors = rng.integers(0, 256, size=(len(unique_values), 3), dtype=np.uint8)

colorised_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
for index, value in enumerate(unique_values):
	colorised_mask[mask == value] = colors[index]

overlay = cv2.addWeighted(rgb, 0.7, colorised_mask, 0.3, 0)

cv2.imwrite("./colorised_mask.png", colorised_mask)
cv2.imwrite("./overlay.png", overlay)


# cv2.imwrite("./mask.png", mask * 100)
