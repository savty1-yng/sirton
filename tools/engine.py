"""Frame compositing engine: erase Hebrew text, draw English text with LTR animations."""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

FPS = 25
W, H = 1920, 1080
FONT_DIR = os.path.join(os.path.dirname(__file__), 'fonts')

PURPLE = (68, 10, 59)
TEAL = (130, 195, 181)
LIGHT = (240, 251, 250)
WHITE = (255, 255, 255)

_fonts = {}


def font(weight, size):
    key = (weight, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(os.path.join(FONT_DIR, f'Rubik-{weight}.ttf'), size)
    return _fonts[key]


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def f(t):
    """Seconds -> frame index."""
    return int(round(t * FPS))


# ---------------------------------------------------------------- items

class Item:
    f0 = 0
    f1 = 0
    layer = 0

    def active(self, i):
        return self.f0 <= i < self.f1


class Erase(Item):
    """Cover a rectangle with background texture taken from clean rows of the same frame,
    or with a solid color."""
    layer = 1

    def __init__(self, f0, f1, rect, band=(30, 280), color=None, feather=8, protect=None):
        self.f0, self.f1, self.rect, self.band, self.color, self.feather = f0, f1, rect, band, color, feather
        self.protect = protect
        x0, y0, x1, y1 = rect
        h, w = y1 - y0, x1 - x0
        m = np.ones((h, w), np.float32)
        fe = feather
        if fe:
            ramp = np.clip((np.arange(w) + 0.5) / fe, 0, 1)
            m *= np.minimum(ramp, ramp[::-1])[None, :]
            ramp = np.clip((np.arange(h) + 0.5) / fe, 0, 1)
            m *= np.minimum(ramp, ramp[::-1])[:, None]
        self.mask = m[..., None]
        if color is None:
            b0, b1 = band
            n = b1 - b0
            k = np.arange(h) % (2 * n)
            self.rows = b0 + np.where(k < n, k, 2 * n - 1 - k)

    def apply(self, i, out, orig):
        x0, y0, x1, y1 = self.rect
        if self.color is None:
            fill = orig[self.rows, x0:x1].astype(np.float32)
        else:
            fill = np.array(self.color, np.float32)[None, None, :]
        reg = out[y0:y1, x0:x1].astype(np.float32)
        mask = self.mask
        if self.protect is not None:
            mask = mask * (1 - self.protect(i, orig[y0:y1, x0:x1]))[..., None]
        out[y0:y1, x0:x1] = (reg * (1 - mask) + fill * mask).astype(np.uint8)


def protect_color(color, tol=70, grow=2):
    def fn(i, reg):
        m = (np.abs(reg.astype(int) - np.array(color)).sum(2) < tol).astype(np.uint8)
        if grow:
            m = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
        return m.astype(np.float32)
    return fn


class Move(Item):
    """Copy a region of the original frame to another position."""
    layer = 2

    def __init__(self, f0, f1, src, dst, feather=0, flip=False, erased=True):
        self.f0, self.f1, self.src, self.dst, self.flip, self.erased = f0, f1, src, dst, flip, erased
        x0, y0, x1, y1 = src
        m = np.ones((y1 - y0, x1 - x0), np.float32)
        if feather:
            for ax, n in ((1, x1 - x0), (0, y1 - y0)):
                r = np.clip((np.arange(n) + 0.5) / feather, 0, 1)
                r = np.minimum(r, r[::-1])
                m *= r[None, :] if ax == 1 else r[:, None]
        self.mask = m[..., None]

    def apply(self, i, out, orig):
        x0, y0, x1, y1 = self.src
        dx, dy = self.dst
        srcimg = self.base if self.erased else orig
        patch = srcimg[y0:y1, x0:x1].astype(np.float32)
        if self.flip:
            patch = patch[:, ::-1]
        h, w = patch.shape[:2]
        reg = out[dy:dy + h, dx:dx + w].astype(np.float32)
        out[dy:dy + h, dx:dx + w] = (reg * (1 - self.mask) + patch * self.mask).astype(np.uint8)


class Sprite(Item):
    """A pre-rendered RGBA image revealed with a left-to-right soft wipe and fade."""
    layer = 3

    def __init__(self, f0, f1, img, xy, appear, dur=10, soft=0.35, fade=True, fadeout=None):
        self.f0, self.f1 = f0, f1
        self.rgba = np.asarray(img).astype(np.float32) / 255.0
        self.xy, self.appear, self.dur, self.soft, self.fade, self.fadeout = xy, appear, dur, soft, fade, fadeout
        h, w = self.rgba.shape[:2]
        self.u = (np.arange(w) / max(w - 1, 1)).astype(np.float32)

    def apply(self, i, out, orig):
        p = (i - self.appear) / self.dur if self.dur else 1.0
        if p <= 0:
            return
        p = min(p, 1.0)
        s = self.soft
        col = np.clip((p * (1 + s) - self.u) / s, 0, 1) if p < 1 else np.ones_like(self.u)
        a = self.rgba[..., 3] * col[None, :]
        if self.fade:
            a = a * (0.35 + 0.65 * smooth(p))
        if self.fadeout is not None and i >= self.fadeout[0]:
            a = a * (1 - smooth((i - self.fadeout[0]) / self.fadeout[1]))
        x, y = self.xy
        h, w = a.shape
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, W), min(y + h, H)
        if x1 <= x0 or y1 <= y0:
            return
        a = a[y0 - y:y1 - y, x0 - x:x1 - x, None]
        rgb = self.rgba[y0 - y:y1 - y, x0 - x:x1 - x, :3] * 255.0
        reg = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = (reg * (1 - a) + rgb * a).astype(np.uint8)


# ---------------------------------------------------------------- text helpers

def text_image(s, weight, size, color, pad=6):
    fnt = font(weight, size)
    l, t, r, b = fnt.getbbox(s)
    asc, desc = fnt.getmetrics()
    w = r - min(l, 0) + 2 * pad
    h = asc + desc + 2 * pad
    img = Image.new('RGBA', (w, h), color + (0,))
    d = ImageDraw.Draw(img)
    d.text((pad - min(l, 0), pad), s, font=fnt, fill=color + (255,))
    return img, pad, asc  # baseline at y = pad + asc


def text_width(s, weight, size):
    fnt = font(weight, size)
    return fnt.getlength(s)


def line_sprite(f0, f1, s, weight, size, color, x, baseline, appear, align='left', dur=10, **kw):
    """x: left edge (align=left), center (align=center) or right edge (align=right)."""
    img, pad, asc = text_image(s, weight, size, color)
    tw = text_width(s, weight, size)
    if align == 'center':
        x = x - tw / 2
    elif align == 'right':
        x = x - tw
    return Sprite(f0, f1, img, (int(round(x - pad)), int(round(baseline - pad - asc))), appear, dur, **kw)


def rich_line(f0, f1, parts, size, x, baseline, appear, align='left', dur=10, **kw):
    """parts: list of (text, weight, color). Rendered as one sprite."""
    widths = [text_width(s, w, size) for s, w, c in parts]
    total = sum(widths)
    fnt0 = font(parts[0][1], size)
    asc, desc = fnt0.getmetrics()
    pad = 6
    img = Image.new('RGBA', (int(total) + 2 * pad + 4, asc + desc + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = pad
    for (s, w, c), tw in zip(parts, widths):
        layer = Image.new('RGBA', img.size, c + (0,))
        ImageDraw.Draw(layer).text((cx, pad), s, font=font(w, size), fill=c + (255,))
        img = Image.alpha_composite(img, layer)
        cx += tw
    if align == 'center':
        x = x - total / 2
    elif align == 'right':
        x = x - total
    return Sprite(f0, f1, img, (int(round(x - pad)), int(round(baseline - pad - asc))), appear, dur, **kw)


def round_rect_sprite(f0, f1, rect, radius, color, appear, dur=10, fade=False, corners=None, **kw):
    x0, y0, x1, y1 = rect
    ss = 4
    img = Image.new('RGBA', ((x1 - x0) * ss, (y1 - y0) * ss), color + (0,))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius=radius * ss, fill=color + (255,), corners=corners)
    img = img.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    return Sprite(f0, f1, img, (x0, y0), appear, dur, fade=fade, **kw)


def block(f0, f1, lines, x, top, appear_times, align='left', dur=10):
    """lines: list of (text, weight, size, color, pitch_before). Returns sprites.
    top: baseline of first line. appear_times: list of frames per line."""
    items = []
    y = top
    for k, (s, wgt, size, color, pitch) in enumerate(lines):
        if k:
            y += pitch
        items.append(line_sprite(f0, f1, s, wgt, size, color, x, y, appear_times[k], align=align, dur=dur))
    return items


# ---------------------------------------------------------------- footer

class Footer(Item):
    """Repaint the teal footer band and write the English footer, following the band as it slides in."""
    layer = 4

    def __init__(self, f0, f1, parts, size=31):
        self.f0, self.f1 = f0, f1
        widths = [text_width(s, w, size) for s, w, c in parts]
        total = sum(widths)
        fnt = font(400, size)
        asc, desc = fnt.getmetrics()
        img = Image.new('RGBA', (W, asc + desc + 12), (0, 0, 0, 0))
        cx = (W - total) / 2
        for (s, w, c), tw in zip(parts, widths):
            layer = Image.new('RGBA', img.size, c + (0,))
            ImageDraw.Draw(layer).text((cx, 6), s, font=font(w, size), fill=c + (255,))
            img = Image.alpha_composite(img, layer)
            cx += tw
        self.img = np.asarray(img).astype(np.float32) / 255.0
        self.asc = asc

    def apply(self, i, out, orig):
        col = orig[990:H, 8].astype(int)
        teal = np.abs(col - np.array(TEAL)).sum(1) < 30
        if not teal.any():
            return
        top = 990 + int(np.argmax(teal))
        y0 = top + 3
        out[y0:H] = np.array(TEAL, np.uint8)
        # Hebrew baseline sits ~51px below band top (band is 77px tall)
        base = top + 51
        ty = base - 6 - self.asc
        a = self.img[..., 3:4]
        h = self.img.shape[0]
        ys, ye = max(ty, y0), min(ty + h, H)
        if ye <= ys:
            return
        a = a[ys - ty:ye - ty]
        rgb = self.img[ys - ty:ye - ty, :, :3] * 255
        reg = out[ys:ye].astype(np.float32)
        out[ys:ye] = (reg * (1 - a) + rgb * a).astype(np.uint8)


class Group(Item):
    """Apply child items, but leave pixels under mask_fn(i, orig) (e.g. foreground graphics) untouched."""
    layer = 1

    def __init__(self, f0, f1, children, mask_fn):
        self.f0, self.f1, self.children, self.mask_fn = f0, f1, children, mask_fn

    def apply(self, i, out, orig):
        tmp = out.copy()
        for c in sorted((c for c in self.children if c.active(i)), key=lambda c: c.layer):
            c.apply(i, tmp, orig)
        m = self.mask_fn(i, orig)
        if m is None:
            out[:] = tmp
            return
        m = m[..., None].astype(np.float32)
        out[:] = (tmp * (1 - m) + out * m).astype(np.uint8)


def compose(i, frame, items):
    orig = frame
    out = frame.copy()
    base = None
    for it in sorted((it for it in items if it.active(i)), key=lambda it: it.layer):
        if isinstance(it, Move):
            if base is None:
                base = out.copy()
            it.base = base
        it.apply(i, out, orig)
    return out


class SpacedText(Item):
    """Centered text whose letter-spacing closes up while it fades in (mirrors the Hebrew tracking animation)."""
    layer = 3

    def __init__(self, f0, f1, s, weight, size, color, cx, baseline, start, dur, spacing0=40):
        self.f0, self.f1 = f0, f1
        self.s, self.weight, self.size, self.color = s, weight, size, color
        self.cx, self.baseline, self.start, self.dur, self.sp0 = cx, baseline, start, dur, spacing0
        self.cache = {}

    def img(self, sp):
        key = int(round(sp * 2))
        if key not in self.cache:
            fnt = font(self.weight, self.size)
            asc, desc = fnt.getmetrics()
            widths = [fnt.getlength(c) for c in self.s]
            total = sum(widths) + sp * (len(self.s) - 1)
            im = Image.new('RGBA', (int(total) + 20, asc + desc + 12), self.color + (0,))
            d = ImageDraw.Draw(im)
            x = 10
            for c, w in zip(self.s, widths):
                d.text((x, 6), c, font=fnt, fill=self.color + (255,))
                x += w + sp
            self.cache[key] = (np.asarray(im).astype(np.float32) / 255.0, total, asc)
        return self.cache[key]

    def apply(self, i, out, orig):
        p = (i - self.start) / self.dur
        if p <= 0:
            return
        e = smooth(p)
        a, total, asc = self.img(self.sp0 * (1 - e))
        x = int(round(self.cx - total / 2 - 10))
        y = int(round(self.baseline - asc - 6))
        h, w = a.shape[:2]
        reg = out[y:y + h, x:x + w].astype(np.float32)
        al = a[..., 3:4] * e
        out[y:y + h, x:x + w] = (reg * (1 - al) + a[..., :3] * 255 * al).astype(np.uint8)


# ---------------------------------------------------------------- tracked replacement

def unblend_alpha(reg, bg, fg, tol=28.0):
    """Per-pixel coverage of colour fg over bg; 0 where the pixel is not on the bg->fg line."""
    bg = np.array(bg, np.float32); fg = np.array(fg, np.float32)
    d = fg - bg
    dd = float((d * d).sum())
    p = reg.astype(np.float32) - bg
    a = (p @ d) / dd
    a = np.clip(a, 0, 1)
    resid = np.sqrt(((p - a[..., None] * d) ** 2).sum(-1))
    a[resid > tol] = 0
    return a


def warp_sprite(out, rgba, cx, cy, s, ang, alpha=1.0, clip=None, anchor=None):
    """Draw rgba (float 0..1) centred (or anchored) at cx,cy with scale s and rotation ang (deg)."""
    h, w = rgba.shape[:2]
    ax, ay = anchor if anchor is not None else (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((ax, ay), ang, s)
    M[0, 2] += cx - ax
    M[1, 2] += cy - ay
    pts = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ M.T
    x0, y0 = np.floor(pts.min(0)).astype(int) - 2
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 2
    if clip is None:
        clip = (0, 0, W, H)
    x0, y0 = max(x0, clip[0]), max(y0, clip[1])
    x1, y1 = min(x1, clip[2]), min(y1, clip[3])
    if x1 <= x0 or y1 <= y0:
        return
    M2 = M.copy(); M2[0, 2] -= x0; M2[1, 2] -= y0
    patch = cv2.warpAffine(rgba, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = patch[..., 3:4] * alpha
    reg = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (reg * (1 - a) + patch[..., :3] * 255 * a).astype(np.uint8)


def render_lines(lines, align='center', gap=None):
    """lines: [(text, weight, size, color)] -> RGBA float image, tight to the glyph boxes vertically by line metrics."""
    imgs = []
    for s, wgt, size, color in lines:
        fnt = font(wgt, size)
        asc, desc = fnt.getmetrics()
        tw = int(np.ceil(fnt.getlength(s))) + 8
        im = Image.new('RGBA', (tw, asc + desc), color + (0,))
        ImageDraw.Draw(im).text((4, 0), s, font=fnt, fill=color + (255,))
        imgs.append((im, size))
    Wd = max(im.width for im, _ in imgs)
    gap = gap if gap is not None else 0
    Ht = sum(im.height for im, _ in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new('RGBA', (Wd, Ht), (0, 0, 0, 0))
    y = 0
    for im, _ in imgs:
        x = (Wd - im.width) // 2 if align == 'center' else (0 if align == 'left' else Wd - im.width)
        canvas.alpha_composite(im, (x, y))
        y += im.height + gap
    a = np.asarray(canvas).astype(np.float32) / 255.0
    # crop to ink
    ys, xs = np.where(a[..., 3] > 0.02)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


class Tracked(Item):
    """Remove Hebrew text of colour fg on a flat colour bg inside a (possibly per-frame) region,
    and draw English that follows the Hebrew blob (position / scale / rotation / visibility)."""
    layer = 2

    def __init__(self, f0, f1, region, bg, fg, sprite, ref, tol=28.0, mode='track',
                 rotate=False, fixed=None, keep_mask=None, min_px=30, wipe=False, scale_by='h', off=(0, 0), smin=0.0, hard=True):
        self.f0, self.f1 = f0, f1
        self.region, self.bg, self.fg, self.sprite = region, bg, fg, sprite
        self.tol, self.mode, self.rotate, self.fixed = tol, mode, rotate, fixed
        self.keep_mask, self.min_px, self.wipe, self.scale_by = keep_mask, min_px, wipe, scale_by
        self.off, self.smin, self.hard = off, smin, hard
        self.ref = ref  # dict filled lazily: count, h, w, angle, cx, cy
        self.refstats = None

    def stats(self, frame, i):
        r = self.region(i, frame) if callable(self.region) else self.region
        if r is None:
            return None, None, None
        x0, y0, x1, y1 = r
        reg = frame[y0:y1, x0:x1]
        a = unblend_alpha(reg, self.bg, self.fg, self.tol)
        m = a > 0.5
        n = int(m.sum())
        if n < self.min_px:
            return r, a, None
        ys, xs = np.nonzero(m)
        st = dict(n=n, x0=xs.min() + x0, x1=xs.max() + x0, y0=ys.min() + y0, y1=ys.max() + y0,
                  cx=xs.mean() + x0, cy=ys.mean() + y0)
        if self.rotate:
            rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            (rw, rh), ang = rect[1], rect[2]
            if rw < rh:
                ang -= 90
            st['ang'] = ang
        return r, a, st

    def set_ref(self, frame):
        if self.sprite is None:
            return
        _, _, st = self.stats(frame, self.ref)
        self.refstats = st
        if st is None:
            raise RuntimeError(f'no reference blob at frame {self.ref}')

    def apply(self, i, out, orig):
        r, a, st = self.stats(orig, i)
        if r is None:
            return
        x0, y0, x1, y1 = r
        if self.keep_mask is not None:
            a = a * (1 - self.keep_mask(i, r))
        reg = out[y0:y1, x0:x1].astype(np.float32)
        d = np.array(self.fg, np.float32) - np.array(self.bg, np.float32)
        res = np.clip(reg - a[..., None] * d, 0, 255)
        if self.hard:
            m = cv2.dilate((a > 0.12).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            if self.keep_mask is not None:
                m &= self.keep_mask(i, r) < 0.5
            res[m] = np.array(self.bg, np.float32)
        out[y0:y1, x0:x1] = res.astype(np.uint8)
        if st is None or self.refstats is None or self.sprite is None:
            return
        R = self.refstats
        vis = min(1.0, st['n'] / R['n'])
        if self.mode == 'fixed':
            cx, cy = self.fixed if self.fixed else (R['cx'] + self.off[0], R['cy'] + self.off[1])
            s, ang = 1.0, 0.0
        elif self.mode == 'enter' and vis < 0.999:
            cx, cy, s, ang = R['cx'] + self.off[0], R['cy'] + self.off[1], 1.0, 0.0
        else:
            if self.scale_by == 'h':
                s = (st['y1'] - st['y0']) / max(R['y1'] - R['y0'], 1)
            else:
                s = np.sqrt(st['n'] / R['n'])
            s = max(s, self.smin)
            cx = st['cx'] + self.off[0] * s
            cy = st['cy'] + self.off[1] * s
            ang = -(st.get('ang', 0) - R.get('ang', 0)) if self.rotate else 0.0
            if self.rotate:
                ang = (ang + 45) % 90 - 45
        if self.mode == 'track':
            vis = min(1.0, st['n'] / (R['n'] * max(s, 1e-3) ** 2))
        spr = self.sprite
        if self.wipe and vis < 0.999:
            u = np.linspace(0, 1, spr.shape[1], dtype=np.float32)
            col = np.clip((vis * 1.3 - u) / 0.3, 0, 1)
            spr = spr.copy(); spr[..., 3] *= col[None, :]
            alpha = 1.0
        else:
            alpha = vis
        warp_sprite(out, spr, cx, cy, s, ang, alpha=alpha)
