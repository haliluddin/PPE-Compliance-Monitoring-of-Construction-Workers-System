# annotate_pose_matplotlib.py
import os, sys, json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if len(sys.argv) < 4:
    print("Usage: python annotate_pose_matplotlib.py images_dir out_json num_keypoints")
    print("Example: python annotate_pose_matplotlib.py test_images /tmp/uploads/gt_pose_job1.json 17")
    sys.exit(1)

images_dir = sys.argv[1]
out_json = sys.argv[2]
num_k = int(sys.argv[3])

imgs = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.png'))])
out = []

print("Instructions:")
print(" Click the top-left and bottom-right corners to create person_bbox (two clicks)")
print(" Then click", num_k, "landmark points in the desired order")
print(" Press Enter when done for that image, or type 's' into console to skip current image")

for fname in imgs:
    path = os.path.join(images_dir, fname)
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.title(f"Annotate {fname}: first two clicks=person_bbox, then {num_k} landmark clicks")
    pts = plt.ginput(2 + num_k, timeout=0)  # blocking until clicks collected
    plt.close()
    if len(pts) < 2 + num_k:
        print("Not enough points clicked, skipping", fname); continue
    x1,y1 = pts[0]; x2,y2 = pts[1]
    person_bbox = [float(min(x1,x2)), float(min(y1,y2)), float(max(x1,x2)), float(max(y1,y2))]
    kps = [[float(p[0]), float(p[1])] for p in pts[2:2+num_k]]
    out.append({"image_id": fname, "person_bbox": person_bbox, "landmarks": kps})
    print("Saved", fname)
# write out
json.dump(out, open(out_json, "w"), indent=2)
print("Wrote", out_json, "with", len(out), "entries")
