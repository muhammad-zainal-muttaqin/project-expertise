import os, re, glob, json
from collections import defaultdict
from PIL import Image, ImageDraw

ROOT953 = "/workspace/SawitMVC-YOLO"
ROOT763 = "/workspace/SawitMVC-Depth-YOLO"
OUT = "/tmp/claude-1001/-workspace/ebcbd941-6775-4113-b727-404085458263/scratchpad"

COLORS = {0: (255, 60, 60), 1: (255, 170, 0), 2: (60, 200, 255), 3: (120, 255, 120)}
NAMES = {0: "B1", 1: "B2", 2: "B3", 3: "B4"}


def load(p):
    out = []
    if not os.path.exists(p):
        return out
    for line in open(p):
        q = line.split()
        if len(q) >= 5:
            out.append((int(q[0]), *map(float, q[1:5])))
    return out


def find953(tree, side):
    for s in ["train", "val", "test"]:
        i = f"{ROOT953}/images/{s}/{tree}_{side}.jpg"
        if os.path.exists(i):
            return i, f"{ROOT953}/labels/{s}/{tree}_{side}.txt"
    return None, None


def find763(tree, side):
    for s in ["train", "valid", "test"]:
        i = f"{ROOT763}/{s}/images/{tree}_{side}.jpg"
        if os.path.exists(i):
            return i, f"{ROOT763}/{s}/labels/{tree}_{side}.txt"
    return None, None


def render(img_path, lbl_path, title, target_h=900):
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    boxes = load(lbl_path)
    for c, cx, cy, w, h in boxes:
        x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
        x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
        d.rectangle([x0, y0, x1, y1], outline=COLORS[c], width=6)
        d.text((x0 + 4, max(0, y0 - 22)), NAMES[c], fill=COLORS[c])
    d.rectangle([0, 0, W, 46], fill=(0, 0, 0))
    d.text((8, 14), f"{title}  |  {len(boxes)} boxes  |  {W}x{H}", fill=(255, 255, 255))
    s = target_h / H
    return im.resize((int(W * s), target_h))


# pick trees with the largest May->July drop
t953, t763 = defaultdict(dict), defaultdict(dict)
for s in ["train", "val", "test"]:
    for f in glob.glob(f"{ROOT953}/labels/{s}/*.txt"):
        b = os.path.basename(f)[:-4]
        m = re.match(r"(.+)_(\d+)$", b)
        t953[m.group(1)][int(m.group(2))] = len(load(f))
for s in ["train", "valid", "test"]:
    for f in glob.glob(f"{ROOT763}/{s}/labels/*.txt"):
        b = os.path.basename(f)[:-4]
        m = re.match(r"(.+)_(\d+)$", b)
        t763[m.group(1)][int(m.group(2))] = len(load(f))

shared = sorted(set(t953) & set(t763))
scored = []
for t in shared:
    for side in [1, 2, 3, 4]:
        a, b = t953[t].get(side), t763[t].get(side)
        if a is not None and b is not None:
            scored.append((a - b, a, b, t, side))
scored.sort(reverse=True)
print("Top per-side drops (May boxes -> July boxes):")
for d_, a, b, t, side in scored[:8]:
    print(f"  {t} side{side}: May={a} Jul={b}")

# median case too
mid = scored[len(scored) // 2]
print(f"Median case: {mid[3]} side{mid[4]}: May={mid[1]} Jul={mid[2]}")

pairs = [scored[0], scored[2], mid]
for k, (_, a, b, tree, side) in enumerate(pairs):
    i1, l1 = find953(tree, side)
    i2, l2 = find763(tree, side)
    left = render(i1, l1, f"MAY 953  {tree}_{side}")
    right = render(i2, l2, f"JULY Depth {tree}_{side}")
    canvas = Image.new("RGB", (left.width + right.width + 20, 900), (30, 30, 30))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + 20, 0))
    p = f"{OUT}/pair_{k}_{tree}_{side}.jpg"
    canvas.save(p, quality=88)
    print("saved", p)
