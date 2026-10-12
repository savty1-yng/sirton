"""Render single frames for review: python3 still.py SRC OUTDIR t1 t2 ...  (STILL=pair stacks original above)."""
import sys, os
import numpy as np, cv2
import scenes, engine

src, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
items = scenes.items()
cap = cv2.VideoCapture(src)


def frame(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    return cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)


def prepare(items, frame):
    for it in items:
        if isinstance(it, engine.Tracked) and it.sprite is not None:
            it.set_ref(frame(it.ref))


prepare(items, frame)
for t in sys.argv[3:]:
    i = int(t[1:]) if t.startswith('f') else int(round(float(t) * 25))
    fr = frame(i)
    out = engine.compose(i, fr, items)
    pair = np.vstack([fr, out]) if os.environ.get('STILL') == 'pair' else out
    path = os.path.join(outdir, f's{t}.png')
    cv2.imwrite(path, cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
    print(path)
