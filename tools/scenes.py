"""Per-scene English layout for "Stage 6 - Part A". Frames are at 25 fps; positions in 1080p pixels."""
import numpy as np
import cv2
from engine import *

R, B, XB, BL = 300, 600, 800, 900   # regular / bold / extra-bold / black
LX = 905                           # left edge of the text column in image+text scenes
PILL = (88, 26, 82)                # purple of pills, bars, bands, circle
BUBBLE_TXT = (248, 242, 248)
TITLE_PURPLE = (69, 10, 66)


def col(f0, f1, lines, x, baselines, times, size=60, color=PURPLE, align='left', dur=10):
    """lines: list of (text, weight) or (text, weight, size)."""
    out = []
    for ln, y, t in zip(lines, baselines, times):
        s, w = ln[0], ln[1]
        sz = ln[2] if len(ln) > 2 else size
        out.append(line_sprite(f0, f1, s, w, sz, color, x, y, t, align=align, dur=dur))
    return out


def text_sprite(lines, align='center', gap=0):
    return render_lines(lines, align=align, gap=gap)


# ------------------------------------------------------------------ helpers for moving graphics

def band_region_title(i, frame):
    """Teal title band of the opening card (rows above the band's bottom edge, right of the logo)."""
    teal = np.abs(frame[:1000:4, ::4].astype(int) - np.array(TEAL)).sum(2) < 30
    rows = np.where(teal.mean(1) > 0.2)[0]
    if len(rows) == 0:
        return None
    cols = np.where(teal[rows].mean(0) > 0.3)[0]
    y1 = rows.max() * 4
    x0, x1 = cols.min() * 4, cols.max() * 4 + 4
    return (max(x0, 350), max(rows.min() * 4, 0), x1, max(y1 - 2, 1))


def pill_region_title(i, frame):
    band = band_region_title(i, frame)
    ytop = band[3] - 10 if band else 300
    sub = frame[ytop:1000:4, 340::4].astype(int)
    m = np.abs(sub - np.array((84, 4, 73))).sum(2) < 40
    ys, xs = np.nonzero(m)
    if len(ys) < 50:
        return None
    x0, x1 = xs.min() * 4 + 340, xs.max() * 4 + 340
    y0, y1 = ys.min() * 4 + ytop, ys.max() * 4 + ytop
    w, h = x1 - x0, y1 - y0
    return (int(x0 + 0.1 * w), int(y0 + 0.08 * h), int(x1 - 0.1 * w), int(y1 - 0.1 * h))


def bubble_region(color):
    def region(i, frame):
        sub = frame[300:1000:2, ::2].astype(int)
        m = (np.abs(sub - np.array(color)).sum(2) < 45).astype(np.uint8)
        if m.sum() < 200:
            return None
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        k = 1 + int(np.argmax(st[1:, 4]))
        x, y, w, h = st[k, :4]
        return (int(x * 2), int(y * 2 + 300), int((x + w) * 2), int((y + h) * 2 + 300))
    return region


class BubbleTracked(Tracked):
    """Tracked text restricted to the inside of one coloured speech bubble."""

    def stats(self, frame, i):
        r = self.region(i, frame) if callable(self.region) else self.region
        if r is None:
            return None, None, None
        x0, y0, x1, y1 = r
        reg = frame[y0:y1, x0:x1]
        bub = (np.abs(reg.astype(int) - np.array(self.bg)).sum(2) < 60).astype(np.uint8)
        if getattr(self, 'grow', 0):
            inside = cv2.dilate(bub, np.ones((self.grow, self.grow), np.uint8)) > 0
            a = unblend_alpha(reg, self.bg, self.fg, self.tol) * inside
            return r, a, None
        inv = (1 - bub).astype(np.uint8)
        # holes = non-bubble regions not connected to the region border (the letters)
        n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        small = st[:, 4] < 0.05 * bub.size
        holes = (~np.isin(lab, list(border))) & small[lab] & (inv > 0)
        inside = (bub > 0) | holes
        inside = cv2.erode(inside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        a = unblend_alpha(reg, self.bg, self.fg, self.tol) * inside
        m = a > 0.5
        nn = int(m.sum())
        if nn < self.min_px:
            return r, a, None
        ys, xs = np.nonzero(m)
        st = dict(n=nn, x0=xs.min() + x0, x1=xs.max() + x0, y0=ys.min() + y0, y1=ys.max() + y0,
                  cx=xs.mean() + x0, cy=ys.mean() + y0)
        if self.rotate:
            rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            (rw, rh), ang = rect[1], rect[2]
            if rw < rh:
                ang -= 90
            st['ang'] = ang
        return r, a, st


BUBBLES = [(112, 160, 204), (224, 116, 75), (239, 194, 57), (219, 92, 110)]


def bubbles_mask(i, frame):
    sub = frame.astype(int)
    m = np.zeros(frame.shape[:2], np.uint8)
    for c in BUBBLES:
        m |= (np.abs(sub - np.array(c)).sum(2) < 60).astype(np.uint8)
    if m.sum() == 0:
        return None
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    inv = 1 - m
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    holes = (st[:, 4] < 40000)[lab] & (inv > 0)
    m = ((m > 0) | holes).astype(np.uint8)
    return cv2.dilate(m, np.ones((5, 5), np.uint8))


def title_mask(i, frame):
    if i >= 569:
        return None
    sub = frame[:1000].astype(int)
    m = np.zeros(frame.shape[:2], np.uint8)
    for c in (TEAL, (84, 4, 73), TITLE_PURPLE):
        m[:1000] |= (np.abs(sub - np.array(c)).sum(2) < 60).astype(np.uint8)
    m[:, :870] = 0
    return cv2.dilate(m, np.ones((5, 5), np.uint8))


def band_region_end(i, frame):
    """The purple band of the closing card."""
    sub = frame[380:1000:4, ::8].astype(int)
    m = np.abs(sub - np.array(PILL)).sum(2) < 40
    rows = np.where(m.mean(1) > 0.6)[0]
    if len(rows) == 0:
        return None
    return (0, rows.min() * 4 + 380 + 2, W, rows.max() * 4 + 380 + 2)


def circle_region(i, frame):
    sub = frame[380:1000:4, 640:1280:4].astype(int)
    m = np.abs(sub - np.array(PILL)).sum(2) < 40
    ys, xs = np.nonzero(m)
    if len(ys) < 50:
        return None
    x0, x1 = xs.min() * 4 + 640, xs.max() * 4 + 640
    y0, y1 = ys.min() * 4 + 380, ys.max() * 4 + 380
    w, h = x1 - x0, y1 - y0
    return (int(x0 + 0.08 * w), int(y0 + 0.08 * h), int(x1 - 0.08 * w), int(y1 - 0.08 * h))


class CircleTracked(Tracked):
    """Text inside the purple circle: only pixels inside the inscribed ellipse are touched."""

    def stats(self, frame, i):
        r, a, st = Tracked.stats(self, frame, i)
        if r is None:
            return r, a, st
        x0, y0, x1, y1 = r
        h, w = a.shape
        yy, xx = np.mgrid[0:h, 0:w]
        # region is the circle bbox shrunk by 8% on each side -> circle radius = 0.5/0.84 of region
        rx, ry = w / 2 / 0.84, h / 2 / 0.84
        inside = ((xx - w / 2) / rx) ** 2 + ((yy - h / 2) / ry) ** 2 < 0.97
        a = a * inside
        m = a > 0.5
        n = int(m.sum())
        if n < self.min_px:
            return r, a, None
        ys, xs = np.nonzero(m)
        st = dict(n=n, x0=xs.min() + x0, x1=xs.max() + x0, y0=ys.min() + y0, y1=ys.max() + y0,
                  cx=xs.mean() + x0, cy=ys.mean() + y0)
        return r, a, st


def items():
    it = []

    # ================================================================ footer (16.4s - end card)
    it.append(Footer(f(16.40), 4979, [
        ('Parents and Words', 700, PURPLE),
        ('  |  ', 300, PURPLE),
        ('A Guidance Program by the Rivka Glustein Method', 300, LIGHT),
        ('  |  ', 300, PURPLE),
        ('Stage 6, Part A', 700, PURPLE),
        ('  |  ', 300, PURPLE),
        ('All Rights Reserved', 300, LIGHT),
    ], size=30))

    # ================================================================ A. copyright notice (0 - 9.0)
    s0, s1 = 0, 225
    it.append(Erase(s0, s1, (370, 368, 1550, 482), color=TEAL, feather=6))
    it.append(Erase(s0, s1, (250, 505, 1700, 925), band=(925, 1000)))
    DP = (75, 25, 72)
    it.append(line_sprite(s0, s1, 'This product is protected by copyright.', XB, 56, DP, 960, 446, 0, align='center', dur=0))
    it += col(s0, s1, [('This product is intended for the purchaser’s use only.', B),
                       ('Transferring or sharing its content with third parties is prohibited.', B)],
              960, [566, 652], [0, 0], size=50, color=DP, align='center', dur=0)
    it += col(s0, s1, [('Copying, duplicating, distributing, reselling,', R),
                       ('or making any commercial use of it of any kind', R),
                       ('without prior permission from the copyright holder is prohibited.', R)],
              960, [744, 820, 896], [0, 0, 0], size=46, color=DP, align='center', dur=0)

    # ================================================================ C. title card (16.4 - 22.7)
    title = text_sprite([('Stage 6', BL, 150, TITLE_PURPLE)])
    part = text_sprite([('Part A', R, 150, TITLE_PURPLE)])
    gapw = 44
    hgt = max(title.shape[0], part.shape[0])
    combo = np.zeros((hgt, title.shape[1] + gapw + part.shape[1], 4), np.float32)
    combo[hgt - title.shape[0]:, :title.shape[1]] = title
    combo[hgt - part.shape[0]:, title.shape[1] + gapw:] = part
    it.append(Tracked(455, 559, band_region_title, TEAL, TITLE_PURPLE, combo, ref=520, mode='enter', wipe=True, off=(0, 6), tol=70))
    it.append(Tracked(559, 569, band_region_title, TEAL, TITLE_PURPLE, combo, ref=520, mode='track', off=(0, 6), tol=70))
    wb = text_sprite([('Word Bank', R, 104, WHITE)])
    it.append(Tracked(480, 559, pill_region_title, (84, 4, 73), (255, 255, 255), wb, ref=520, mode='track', tol=70, off=(0, 4), smin=1.0))
    it.append(Tracked(559, 569, pill_region_title, (84, 4, 73), (255, 255, 255), wb, ref=520, mode='track', tol=70, off=(0, 4)))

    # ================================================================ D. greeting (22.4 - 32.0)
    s0, s1 = 559, 800
    it.append(Group(559, s1, [Erase(559, s1, (880, 280, 1840, 830))] +
                    col(s0, s1, [('Hello, dear parents,', R), ('and welcome to Stage 6.', R)],
                        LX, [354, 450], [f(22.40), f(22.52)]), title_mask))
    it += col(s0, s1, [('Here, we go a step deeper', B, 64), ('in the “Parents and Words”', B, 64), ('program.', B, 64)],
              LX, [608, 702, 796], [f(28.24), f(28.40), f(28.52)])

    # ================================================================ E. trigger (32.0 - 37.8)
    s0, s1 = 800, 944
    it.append(Erase(s0, s1, (880, 470, 1840, 870)))
    it.append(line_sprite(s0, s1, 'We start from the understanding', B, 56, PURPLE, LX, 552, f(32.20)))
    bars = ['that there is always', 'some kind of trigger', 'behind a child’s behavior.']
    for k, s in enumerate(bars):
        top = 590 + 92 * k
        w = int(text_width(s, B, 54))
        t = f(32.24) + 4 * k
        it.append(round_rect_sprite(s0, s1, (LX - 27, top, LX + w + 30, top + 76), 0, TEAL, t, dur=8))
        it.append(line_sprite(s0, s1, s, B, 54, PURPLE, LX, top + 57, t + 3, dur=8))

    # ================================================================ F. no bad child (37.8 - 46.3)
    s0, s1 = 944, 1159
    it.append(Erase(s0, s1, (880, 330, 1840, 960)))
    it += col(s0, s1, [('We’ve already seen that a child', B), ('doesn’t act out for no reason.', B)],
              LX, [412, 508], [f(37.88), f(38.00)])
    it += col(s0, s1, [('As we all know, there’s no such', R), ('thing as a bad child — only a child', R),
                       ('who feels bad, who is struggling', R), ('for some reason.', R)],
              LX, [604, 698, 792, 886], [f(41.92), f(42.20), f(42.72), f(42.96)], size=56)

    # ================================================================ G. how are we to know (46.3 - 53.4)
    s0, s1 = 1159, 1336
    it.append(Erase(s0, s1, (880, 370, 1840, 800)))
    it += col(s0, s1, [('And when a child struggles,', R), ('and he himself doesn’t know', R), ('how to explain it,', R)],
              LX, [448, 540, 632], [f(46.28), f(46.40), f(46.52)], size=58)
    it.append(line_sprite(s0, s1, 'How are we to know?', 700, 84, PURPLE, LX, 742, f(47.70), dur=42))

    # ================================================================ H. baby crying (53.4 - 59.9)
    s0, s1 = 1336, 1497
    it.append(Erase(s0, s1, (860, 300, 1840, 800)))
    it.append(line_sprite(s0, s1, 'It’s interesting —', B, 54, PURPLE, LX, 372, f(53.60)))
    it += col(s0, s1, [('when a mother hears her baby crying,', R), ('she understands this “baby language”', R),
                       ('and immediately sets out to find out', R)],
              LX, [462, 552, 642], [f(53.68), f(53.80), f(53.88)], size=52)
    it.append(line_sprite(s0, s1, 'what’s bothering him.', 700, 78, PURPLE, LX, 752, f(53.96)))

    # ================================================================ I. maybe... (59.9 - 65.0)
    s0, s1 = 1497, 1625
    it.append(Erase(s0, s1, (880, 290, 1840, 790)))
    it += col(s0, s1, [('Maybe he needs to burp?', R), ('Maybe he needs a diaper change?', R), ('Maybe he’s cold or hot?', R),
                       ('Maybe he’s tired?', R), ('Or hungry?', R)],
              LX, [366, 462, 558, 654, 750], [f(60.04), f(60.12), f(60.20), f(60.32), f(60.40)], size=56)

    # ================================================================ J. pediatrician (65.0 - 71.8)
    s0, s1 = 1625, 1795
    it.append(Erase(s0, s1, (880, 340, 1840, 740)))
    it += col(s0, s1, [('And if she can’t figure it out,', R), ('she’ll go to the pediatrician', R),
                       ('to check if something is hurting him,', R), ('or if he’s not feeling well.', R)],
              LX, [414, 510, 606, 702], [f(65.12), f(65.20), f(65.32), f(65.44)], size=54)

    # ================================================================ K. as children grow (71.8 - 80.9)
    s0, s1 = 1795, 2022
    it.append(Erase(s0, s1, (370, 390, 1600, 588)))
    it.append(line_sprite(s0, s1, 'But as children grow and begin to speak,', B, 56, PURPLE, 960, 458, f(71.96), align='center'))
    it.append(line_sprite(s0, s1, 'we search less —', R, 60, PURPLE, 960, 556, f(74.00), align='center'))
    it.append(Erase(1898, s1, (382, 602, 1538, 806), color=TEAL, feather=4))
    it += col(1898, s1, [('and rely more on what', R), ('the child tells us directly.', R)],
              960, [680, 776], [1899, 1902], size=62, color=(236, 248, 245), align='center', dur=12)

    # ================================================================ L. too quickly (80.9 - 88.6)
    s0, s1 = 2022, 2216
    it.append(Erase(s0, s1, (380, 300, 1560, 610)))
    it += col(s0, s1, [('Yet sometimes,', R), ('we rely on his words alone', R), ('a little too quickly.', R)],
              960, [378, 474, 570], [f(81.08), f(81.16), f(81.32)], align='center')

    # ================================================================ M. preschool (88.6 - 97.0)
    s0, s1 = 2216, 2424
    it.append(Erase(s0, s1, (880, 290, 1840, 790)))
    it += col(s0, s1, [('So if a child doesn’t share', R), ('what happened in preschool,', R),
                       ('we may rush to address the emotions', R), ('instead of pausing for a moment', R),
                       ('and thinking developmentally:', R)],
              LX, [362, 458, 554, 650, 746], [f(88.84), f(88.88), f(92.80), f(94.12), f(95.44)], size=52, dur=13)

    # ================================================================ N. a year or two ago (97.0 - 108.4)
    s0, s1 = 2424, 2711
    it.append(Erase(s0, s1, (880, 350, 1840, 860)))
    it += col(s0, s1, [('What would we have done', B), ('just a year or two ago?', B)],
              LX, [428, 522], [f(97.08), f(97.16)])
    it += col(s0, s1, [('Let’s keep exploring a little more', R), ('and try to understand what’s really', R),
                       ('going on with the child.', R)],
              LX, [620, 714, 808], [f(100.24), f(101.80), f(102.10)], size=54, dur=13)

    # ================================================================ O. pill question (108.4 - 113.5)
    s0, s1 = 2711, 2838
    it.append(Erase(s0, s1, (190, 470, 1730, 612), color=PILL, feather=4))
    it.append(line_sprite(s0, s1, 'So how do we know what the real problem is?', XB, 64, TEAL, 960, 560, s0, align='center', dur=0))

    # ================================================================ P. instead of the result (113.5 - 125.8)
    s0, s1 = 2838, 3145
    LXP = 470
    it.append(Erase(s0, s1, (440, 270, 1560, 960)))
    hw = int(text_width('Instead of describing the result:', XB, 56))
    it.append(round_rect_sprite(s0, s1, (LXP - 6, 320, LXP + hw + 34, 356), 18, TEAL, f(113.64), dur=10))
    it.append(line_sprite(s0, s1, 'Instead of describing the result:', XB, 56, PURPLE, LXP + 8, 338, f(113.70), dur=10))
    items_p = ['acting wild,', 'throwing things,', 'making a mess,', 'hitting,', 'sitting quietly in the corner,',
               'and not sharing what happened in school —']
    tops = [399, 494, 588, 682, 777, 871]
    times = [f(116.44), f(116.84), f(117.24), f(117.60), f(118.00), f(118.40)]
    for s, top, t in zip(items_p, tops, times):
        it.append(Move(s0, s1, (1476, top - 4, 1530, top + 46), (LXP, top - 4), erased=False))
        it.append(line_sprite(s0, s1, s, R, 58, PURPLE, LXP + 80, top + 37, t))

    # ================================================================ Q. hearts (125.8 - 132.9)
    s0, s1 = 3145, 3322
    it.append(Erase(s0, s1, (440, 300, 1500, 410)))
    it.append(line_sprite(s0, s1, 'we need to open our eyes and pay attention:', R, 60, PURPLE, 960, 376, f(126.00), align='center'))
    it.append(Erase(s0, s1, (1070, 545, 1470, 780), feather=10, protect=protect_color(TEAL)))
    it.append(Erase(s0, s1, (430, 545, 840, 780), feather=10, protect=protect_color(TEAL)))
    # swap the hearts so the first one drawn sits on the left
    it.append(Move(s0, s1, (973, 425, 1601, 965), (345, 425), feather=5))
    it.append(Move(s0, s1, (345, 425, 973, 965), (973, 425), feather=5))
    it += col(s0, s1, [('What happens', R), ('before the', R), ('outburst', R)],
              637, [612, 690, 768], [f(128.30), f(128.42), f(128.54)], size=56, align='center')
    it += col(s0, s1, [('And what', R), ('helps it', R), ('calm down', R)],
              1265, [612, 690, 768], [f(130.20), f(130.32), f(130.44)], size=56, align='center')

    # ================================================================ R. tools / circle (132.9 - 142.5)
    s0, s1 = 3322, 3564
    it.append(Erase(s0, s1, (380, 235, 1560, 395)))
    it += col(s0, s1, [('That way, we as parents can use the tools', R),
                       ('we’re learning in the “Parents and Words” program', R)],
              960, [294, 372], [f(133.08), f(133.28)], size=50, align='center')
    circ = text_sprite([('to help', R, 48, BUBBLE_TXT), ('our children overcome', R, 48, BUBBLE_TXT),
                        ('the daily frustrations', R, 48, BUBBLE_TXT), ('that cause them', R, 48, BUBBLE_TXT),
                        ('distress,', R, 48, BUBBLE_TXT)], gap=16)
    it.append(CircleTracked(3430, s1, circle_region, PILL, BUBBLE_TXT, circ, ref=3500, mode='track', tol=70))

    # ================================================================ S. be children (142.5 - 151.7)
    s0, s1 = 3564, 3794
    LXS = 610
    it.append(Erase(s0, s1, (580, 280, 1640, 860)))
    it += col(s0, s1, [('and use their energy instead', R), ('to simply be children —', R)],
              LXS, [352, 448], [f(143.04), f(143.16)])
    for k, (s, t) in enumerate(zip(['happy', 'calm', 'and content.'], [f(144.88), f(145.16), f(145.40)])):
        top = [516, 636, 752][k]
        it.append(Move(s0, s1, (1536, top - 2, 1600, top + 62), (LXS, top - 2), erased=False))
        it.append(line_sprite(s0, s1, s, 700, 66, PURPLE, LXS + 92, top + 54, t))

    # ================================================================ T. question bars (151.7 - 155.1)
    s0, s1 = 3812, 3879
    for k, s in enumerate(['What', 'happens when', 'a child has', 'a limited', 'vocabulary?']):
        top = [280, 401, 522, 643, 764][k]
        it.append(Erase(s0, s1, (716, top + 2, 1204, top + 78), color=PILL, feather=2))
        it.append(line_sprite(s0, s1, s, XB, 60, (246, 242, 246), 736, top + 61, s0, dur=0))

    # ================================================================ U. speech bubbles (155.1 - 163.9)
    s0, s1 = 3879, 4097
    LXU = 900
    it.append(Group(s0, s1, [
        Erase(s0, s1, (740, 240, 1830, 330)),
        Erase(s0, s1, (1080, 330, 1830, 445)),
    ], bubbles_mask))
    for c in BUBBLES:
        bt = BubbleTracked(4080, s1, (740, 240, 1830, 445), c, PURPLE, None, ref=None, tol=70)
        bt.grow = 11
        it.append(bt)
    it.append(line_sprite(s0, s1, 'Very often, we’ll hear words like:', R, 58, PURPLE, LXU, 306, s0, dur=8))
    it.append(line_sprite(s0, s1, 'without any explanation or detail.', R, 58, PURPLE, LXU, 402, f(160.00), dur=40))
    for color, words, wgt, size, rot, ref in [
            ((112, 160, 204), ['Good'], XB, 84, True, 4030),
            ((224, 116, 75), ['Not good'], XB, 76, True, 4030),
            ((239, 194, 57), ['I want'], XB, 82, True, 4030),
            ((219, 92, 110), ['I don’t', 'want'], R, 46, False, 4030)]:
        spr = text_sprite([(w, wgt, size, WHITE) for w in words], gap=-4)
        it.append(BubbleTracked(s0, 4099, bubble_region(color), color, (255, 255, 255), spr, ref=ref,
                                mode='track', rotate=rot, tol=70, min_px=12))

    # ================================================================ V. embarrassed (163.9 - 170.3)
    s0, s1 = 4097, 4257
    it.append(Erase(s0, s1, (880, 350, 1840, 830)))
    it += col(s0, s1, [('And that leaves us,', R), ('as parents, feeling confused,', R), ('and sometimes unsure', R),
                       ('how to respond', R), ('in the moment.', R)],
              LX, [422, 516, 610, 704, 798], [f(163.92), f(164.08), f(164.20), f(164.28), f(164.36)], size=58)

    # ================================================================ W. socially (170.3 - 176.2)
    s0, s1 = 4257, 4406
    it.append(Erase(s0, s1, (880, 430, 1840, 730)))
    it += col(s0, s1, [('We can see and hear', R), ('that they’re struggling socially,', R), ('but we don’t understand why.', R)],
              LX, [506, 600, 694], [f(170.44), f(170.60), f(170.68)])

    # ================================================================ X. social skills (176.2 - 180.3)
    s0, s1 = 4406, 4507
    it.append(Erase(s0, s1, (880, 430, 1840, 640)))
    it += col(s0, s1, [('So before we rush to put them', R), ('into a social skills program,', R)],
              LX, [506, 600], [f(176.40), f(176.60)])

    # ================================================================ Y. basket (180.3 - 186.3)
    s0, s1 = 4507, 4658
    it.append(Erase(s0, s1, (380, 385, 1560, 690)))
    it += col(s0, s1, [('Perhaps we can simply give them', R), ('a set of useful words that will help', R),
                       ('them open the door to friendships.', R)],
              960, [464, 558, 652], [f(180.28), f(180.40), f(180.54)], align='center')

    # ================================================================ Z. next lesson (186.3 - 199.1)
    s0, s1 = 4658, 4979
    it.append(Erase(s0, s1, (380, 270, 1560, 455)))
    it += col(s0, s1, [('In the next lesson, you’ll receive', R), ('a powerful tool called', R)],
              960, [328, 422], [4660, 4668], align='center', dur=14)
    psl = text_sprite([('“Practical Social Language”', XB, 78, TEAL)])
    it.append(Tracked(4730, 4803, band_region_end, PILL, TEAL, psl, ref=4790, mode='enter', wipe=True, tol=70))
    wbk = text_sprite([('“Word Bank”', XB, 92, TEAL)])
    it.append(Tracked(4803, s1, band_region_end, PILL, TEAL, wbk, ref=4900, mode='track', scale_by='n', tol=70))
    it.append(Erase(4790, 4940, (352, 712, 1600, 790)))
    it.append(Erase(4940, s1, (382, 712, 1600, 790)))
    it.append(Erase(4940, s1, (344, 712, 392, 758), feather=4))
    it.append(SpacedText(4790, s1, 'as the children who use it call it', R, 50, PURPLE, 960, 762, 4800, 34, spacing0=14))
    it.append(Erase(4880, s1, (420, 840, 1500, 960)))
    it.append(SpacedText(4880, s1, 'Wishing you much success.', XB, 66, (75, 8, 64), 960, 915, 4905, 40, spacing0=26))

    return it
