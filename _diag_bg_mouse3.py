# -*- coding: utf-8 -*-
"""后台鼠标注入严格探测 v2：A/B/A 三拍差分 + ROI 局部统计 + 动画噪声基线。

每次手法执行 P1 -> P2 -> 回 P1 三拍截图；若游戏光标跟随：
  ROI(P2) 在 [P1拍 vs P2拍] 出现变化（光标到达）
  ROI(P1) 在 [P2拍 vs P1拍] 出现变化（光标回来到达）
对照基线：无输入连续三拍同 ROI 的自然变化（动画噪声）。
纯后台测试期间真实光标被挪出游戏窗口，结束后还原。
"""
import ctypes
import time

import cv2
import numpy as np
import win32api
import win32con
import win32gui
import win32process

GAME_TITLE = "Moonlight"
PW_RENDERFULLCONTENT = 2
ROI = 70  # 探测点邻域半宽。
NOISE_FACTOR = 3.0  # 判定阈值 = 基线噪声 * 系数 + 绝对下限。
NOISE_FLOOR = 25


def find_game():
    hits = []
    def cb(hwnd, _):
        t = win32gui.GetWindowText(hwnd)
        if t and GAME_TITLE in t and win32gui.IsWindowVisible(hwnd):
            l, tp, r, b = win32gui.GetClientRect(hwnd)
            hits.append(((r - l) * (b - tp), hwnd))
    win32gui.EnumWindows(cb, None)
    return max(hits)[1] if hits else None


def capture(hwnd):
    l, t, r, b = win32gui.GetClientRect(hwnd)
    w, h = r - l, b - t
    hdc = win32gui.GetDC(hwnd)
    mdc = win32gui.CreateCompatibleDC(hdc)
    bmp = win32gui.CreateCompatibleBitmap(hdc, w, h)
    win32gui.SelectObject(mdc, bmp)
    result = ctypes.windll.user32.PrintWindow(hwnd, mdc, PW_RENDERFULLCONTENT)
    win32gui.SelectObject(mdc, 0)
    from ctypes import wintypes
    class BITMAPHEADER(ctypes.Structure):
        _fields_ = [("biSize", wintypes.UINT), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                    ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.UINT),
                    ("biSizeImage", wintypes.UINT), ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                    ("biClrUsed", wintypes.UINT), ("biClrImportant", wintypes.UINT)]
    bmi = BITMAPHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPHEADER)
    bmi.biWidth, bmi.biHeight, bmi.biPlanes, bmi.biBitCount, bmi.biCompression = w, -h, 1, 32, 0
    buf = ctypes.create_string_buffer(w * h * 4)
    ok = ctypes.windll.gdi32.GetDIBits(int(hdc), int(bmp), 0, h, buf, ctypes.byref(bmi), 0)
    win32gui.DeleteObject(bmp)
    win32gui.DeleteDC(mdc)
    win32gui.ReleaseDC(hwnd, hdc)
    if not result or not ok:
        return None
    return np.frombuffer(buf.raw, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()


def roi_diff(a, b, pt):
    """只统计探测点邻域的变化像素数。"""
    if a is None or b is None:
        return -1
    h, w = a.shape[:2]
    x0 = max(0, pt[0] - ROI); x1 = min(w, pt[0] + ROI)
    y0 = max(0, pt[1] - ROI); y1 = min(h, pt[1] + ROI)
    d = cv2.absdiff(a[y0:y1, x0:x1], b[y0:y1, x0:x1]).max(axis=2)
    return int((d > 25).sum())


def post_move(hwnd, x, y):
    win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, win32api.MAKELONG(x, y))


def send_move(hwnd, x, y):
    ctypes.windll.user32.SendMessageTimeoutW(hwnd, win32con.WM_MOUSEMOVE, 0, win32api.MAKELONG(x, y), 0, 500, None)


def activate_forgery(hwnd):
    user32 = ctypes.windll.user32
    tid = win32api.GetCurrentThreadId()
    _, target_tid = win32process.GetWindowThreadProcessId(hwnd)
    attached = False
    try:
        attached = user32.AttachThreadInput(tid, target_tid, True)
        user32.SetActiveWindow(hwnd)
        user32.SetFocus(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(tid, target_tid, False)


def real_move_factory(hwnd):
    def move(x, y):
        sx, sy = win32gui.ClientToScreen(hwnd, (x, y))
        win32api.SetCursorPos((sx, sy))
    return move


def probe(name, mover, settle=0.3):
    """P1 -> P2 -> P1 三拍；返回 (到达P2变化, 返回P1变化)。"""
    mover(*P1); time.sleep(settle); c1 = capture(hwnd)
    mover(*P2); time.sleep(settle); c2 = capture(hwnd)
    mover(*P1); time.sleep(settle); c3 = capture(hwnd)
    arrive = roi_diff(c1, c2, P2)  # 光标应到达 P2。
    back = roi_diff(c2, c3, P1)    # 光标应回到 P1。
    return arrive, back


hwnd = find_game()
assert hwnd, "game window not found"
if win32gui.IsIconic(hwnd):
    print("restoring minimized window...")
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    time.sleep(1.5)
cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
W, H = cr - cl, cb - ct
P1, P2 = (int(W * 0.30), int(H * 0.45)), (int(W * 0.70), int(H * 0.45))
print(f"hwnd={hwnd} client={W}x{H} P1={P1} P2={P2}")

# 真实光标挪出游戏窗口（放到屏幕右上角），避免污染后台手法测试。
orig_cursor = win32gui.GetCursorPos()
scr_w = win32api.GetSystemMetrics(0)
win32api.SetCursorPos((scr_w - 5, 5))
time.sleep(0.3)

def force_foreground(hwnd):
    """把窗口带到前台：AttachThreadInput 前台窃取，失败退回最小化/还原。"""
    user32 = ctypes.windll.user32
    fg = user32.GetForegroundWindow()
    tid = win32api.GetCurrentThreadId()
    _, fg_tid = win32process.GetWindowThreadProcessId(fg) if fg else (0, 0)
    _, target_tid = win32process.GetWindowThreadProcessId(hwnd)
    attached = False
    try:
        if fg_tid:
            attached = user32.AttachThreadInput(tid, fg_tid, True)
        user32.AttachThreadInput(tid, target_tid, True)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if fg_tid:
            user32.AttachThreadInput(tid, fg_tid, False)
        user32.AttachThreadInput(tid, target_tid, False)
    if user32.GetForegroundWindow() != hwnd:  # 仍未到前台：最小化/还原强推。
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        time.sleep(0.2)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        time.sleep(0.3)


# 渲染活性：置前后最多等 8 秒，直到无输入连续两帧全局有变化（游戏被遮挡/未聚焦会暂停渲染）。
force_foreground(hwnd)
alive = False
prev = capture(hwnd)
for _ in range(16):
    time.sleep(0.5)
    cur = capture(hwnd)
    if prev is not None and cur is not None and int((cv2.absdiff(prev, cur).max(axis=2) > 25).sum()) > 50:
        alive = True
        break
    prev = cur
print("rendering:", alive)
if not alive:
    print("!! 游戏画面 8 秒内完全静止（未渲染）。请让游戏窗口可见且正在活动画面，再重跑本探测。")

f1 = capture(hwnd); time.sleep(0.3); f2 = capture(hwnd); time.sleep(0.3); f3 = capture(hwnd)
noise_p1 = max(roi_diff(f1, f2, P1), roi_diff(f2, f3, P1))
noise_p2 = max(roi_diff(f1, f2, P2), roi_diff(f2, f3, P2))
noise = max(noise_p1, noise_p2)
print(f"animation noise in ROIs: P1={noise_p1} P2={noise_p2} threshold={max(noise * NOISE_FACTOR, NOISE_FLOOR):.0f}")
if noise < 3:
    print("!! 画面几乎不变（游戏未渲染或静止），探测意义有限，继续但注意判读。")

threshold = max(noise * NOISE_FACTOR, NOISE_FLOOR)

# 纯后台手法。
results = {}
a, b = probe("T0 post", lambda x, y: post_move(hwnd, x, y))
results["T0 PostMessage"] = (a, b)
activate_forgery(hwnd)
a, b = probe("T1 act+post", lambda x, y: post_move(hwnd, x, y))
results["T1 激活伪造+Post"] = (a, b)
activate_forgery(hwnd)
a, b = probe("T2 send", lambda x, y: send_move(hwnd, x, y))
results["T2 SendMessage"] = (a, b)

# 阳性对照：真实光标。
real = real_move_factory(hwnd)
a, b = probe("R0 real", real)
results["R0 真实光标(对照)"] = (a, b)

win32api.SetCursorPos(orig_cursor)
print("cursor restored to", orig_cursor)
print()
for name, (arrive, back) in results.items():
    verdict = "生效" if (arrive >= threshold and back >= threshold) else "无反应"
    print(f"{name:24s} 到达P2={arrive:5d} 回到P1={back:5d} 阈值={threshold:.0f} => {verdict}")
