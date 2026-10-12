"""Render the English version: python3 render.py SRC OUT.mp4 [workers]"""
import os, sys, subprocess, tempfile
from multiprocessing import Process
import numpy as np, cv2

W, H, FPS = 1920, 1080, 25


def worker(src, a, b, path):
    import scenes, engine
    items = scenes.items()
    cap = cv2.VideoCapture(src)

    def frame(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        return cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)

    for it in items:
        if isinstance(it, engine.Tracked) and it.sprite is not None:
            it.set_ref(frame(it.ref))
    dec = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', src, '-vf',
                            f'trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS',
                            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                            '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p',
                            '-x264-params', 'keyint=50', path], stdin=subprocess.PIPE)
    n = W * H * 3
    i = a
    while i < b:
        buf = dec.stdout.read(n)
        if len(buf) < n:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        out = engine.compose(i, fr, items)
        enc.stdin.write(out.tobytes())
        i += 1
        if (i - a) % 250 == 0:
            print(f'[{a}-{b}] {i}', flush=True)
    enc.stdin.close(); enc.wait(); dec.wait()


if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    total = int(cv2.VideoCapture(src).get(cv2.CAP_PROP_FRAME_COUNT))
    tmp = tempfile.mkdtemp()
    edges = np.linspace(0, total, nw + 1).astype(int)
    procs, parts = [], []
    for k in range(nw):
        p = os.path.join(tmp, f'part{k}.mp4'); parts.append(p)
        pr = Process(target=worker, args=(src, edges[k], edges[k + 1], p)); pr.start(); procs.append(pr)
    for pr in procs:
        pr.join()
        if pr.exitcode:
            sys.exit('worker failed')
    lst = os.path.join(tmp, 'list.txt')
    open(lst, 'w').write(''.join(f"file '{p}'\n" for p in parts))
    subprocess.check_call(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-i', src,
                           '-map', '0:v', '-map', '1:a', '-c', 'copy', '-movflags', '+faststart', out])
    print('done', out)
