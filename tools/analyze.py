"""Find Hebrew text lines in a scene and when each one appears."""
import sys, json
import numpy as np, cv2

S = sys.argv[1]
g = np.load(f'{S}/gray.npy', mmap_mode='r')  # half-res gray frames
cap = cv2.VideoCapture(f'{S}/src.mp4')


def lines_in(frame_idx, box, thr=170, min_gap=6):
    x0, y0, x1, y1 = [v // 2 for v in box]
    r = g[frame_idx, y0:y1, x0:x1] < thr
    rows = r.sum(1) > 1
    out, s = [], None
    for y, v in enumerate(list(rows) + [False]):
        if v and s is None: s = y
        if not v and s is not None:
            if y - s >= 3: out.append([s, y])
            s = None
    merged = []
    for a, b in out:
        if merged and a - merged[-1][1] < min_gap // 2 + 1: merged[-1][1] = b
        else: merged.append([a, b])
    res = []
    for a, b in merged:
        cols = np.where(r[a:b].any(0))[0]
        res.append(((x0 + cols[0]) * 2, (y0 + a) * 2, (x0 + cols[-1] + 1) * 2, (y0 + b) * 2))
    return res


def timing(box, f0, f1, thr=170):
    x0, y0, x1, y1 = [v // 2 for v in box]
    ink = np.array([(g[i, y0:y1, x0:x1] < thr).sum() for i in range(f0, f1)], float)
    fin = ink.max()
    if fin == 0: return None
    a = f0 + int(np.argmax(ink > 0.1 * fin)); b = f0 + int(np.argmax(ink > 0.9 * fin))
    return a, b


def color(frame_idx, box):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx); f = cap.read()[1]
    x0, y0, x1, y1 = box
    c = f[y0:y1, x0:x1].reshape(-1, 3); gg = c.mean(1)
    dark = c[gg < np.percentile(gg, 8)]
    return '#%02x%02x%02x' % tuple(int(v) for v in dark.mean(0)[::-1])


if __name__ == '__main__':
    t0, t1, ref = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    box = [int(v) for v in sys.argv[5].split(',')]
    f0, f1, fr = int(t0 * 25), int(t1 * 25), int(ref * 25)
    for L in lines_in(fr, box):
        tm = timing(L, f0, f1)
        print(L, 'h=%d' % (L[3] - L[1]), 'in=%.2f full=%.2f' % (tm[0] / 25, tm[1] / 25) if tm else '-', color(fr, L))
