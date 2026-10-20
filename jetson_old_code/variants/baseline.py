"""baseline.py - so tay cac trang thai DA CHAY, de khong bao gio mat lai.

    python3 baseline.py list
    python3 baseline.py save <thu_muc_lan_chay> "<nhan>"
    python3 baseline.py diff <nhan>
    python3 baseline.py restore <nhan>

Ton tai vi 2026-09-11: nguoi dung noi "lúc trước xe đã chạy thẳng được", va
khong ai chi ra duoc CHINH XAC luc do la bo thong so nao. Phai doi chieu tay
17 lan chay moi tim ra, roi lai phat hien lan do thang vi mot ly do khong ai
muon: chenh lech banh chi 0.15 tick tren moi don vi steer, tuc xe gan nhu
KHONG LAI DUOC nen khong the lech.

Vi vay so tay nay luu CA HAI: bo thong so, va ket qua do duoc cua no. Mot
baseline khong kem so do thi chi la mot dong so vo nghia - khong biet no
"tot" theo nghia nao.

Luu vao baselines/<nhan>/ gom:
  Confg.py        ban config chinh xac cua lan chay do
  firmware.txt    cac hang so firmware dang duoc nap luc do
  meta.json       so do ket qua + commit git + duong dan lan chay goc
"""
import csv
import glob
import json
import os
import re
import shutil
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(HERE, "baselines")
FIRMWARE = os.path.join(HERE, "motor_test_bts7960", "src", "main.cpp")

ENC_RE = re.compile(
    r"ENC,(-?\d+),(-?\d+),DELTA,(-?\d+),(-?\d+),TARGET,(-?\d+),(-?\d+),"
    r"PWM,(-?\d+),(-?\d+),STEER,(-?\d+),SPEED,(-?\d+),MODE,(\w)"
)
# Cac hang so firmware that su quyet dinh hanh vi lai. Doc tu file .cpp chu
# khong tu doan - 2026-09-10 da mot lan phan tich sai file firmware.
FW_KEYS = ("SPEED_REFERENCE", "PWM_MAX", "PWM_MIN", "BASE_PWM",
           "TARGET_TICKS_AT_REFERENCE", "MAX_WHEEL_TARGET_TICKS",
           "STEER_GAIN_PERCENT", "STEER_COMMAND_MAX", "SPEED_COMMAND_MAX",
           "KP", "KI", "INTEGRAL_LIMIT", "RIGHT_WHEEL_PWM_TRIM")
CFG_KEYS = ("PID_KP", "PID_KI", "PID_KD", "PID_MAX_OUTPUT",
            "PID_INTEGRAL_LIMIT", "PID_STEP_LIMIT", "BASE_SPEED", "MIN_SPEED",
            "RECOVERY_SPEED", "STEER_CENTER_X", "STEER_TRIM",
            "HEADING_ERROR_WEIGHT", "LANE_LOST_MAX_FRAMES",
            "LANE_ERROR_SMOOTHING_ALPHA", "LANE_LOST_STEER_DECAY")


def read_consts(path, keys):
    out = {}
    if not os.path.exists(path):
        return out
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
    except IOError:
        return out
    for k in keys:
        m = re.search(r"^\s*(?:#define\s+|(?:const\s+)?\w+\s+)?%s\s*[= ]\s*"
                      r"([-\d.]+)" % re.escape(k), src, re.M)
        if m:
            out[k] = m.group(1)
    return out


def measure(run_dir):
    """So do ket qua cua mot lan chay. Khong co so do thi baseline vo nghia."""
    m = {}
    tel = glob.glob(os.path.join(run_dir, "patrol_telem_*.csv"))
    if tel:
        rows = list(csv.DictReader(open(tel[0])))
        if rows:
            m["frames"] = len(rows)
            t = [float(r["t"]) for r in rows]
            m["duration_s"] = round(t[-1] - t[0], 1)
            gaps = [b - a for a, b in zip(t, t[1:]) if b > a]
            if gaps:
                m["fps"] = round(1.0 / st.median(gaps), 2)
            L = [r for r in rows if r["src"] == "LINE"]
            m["line_pct"] = round(100.0 * len(L) / len(rows))
            if L:
                e = [float(r["raw_err"]) for r in L]
                m["err_median"] = round(st.median(e))
                m["err_abs_median"] = round(st.median([abs(v) for v in e]))
                m["err_abs_p90"] = round(sorted(abs(v) for v in e)[int(.9 * len(e))])
                m["err_right_frac"] = round(
                    sum(1 for v in e if v > 0) / float(len(e)), 2)
    enc = []
    bl = os.path.join(run_dir, "bridge.log")
    if os.path.exists(bl):
        for ln in open(bl, errors="replace"):
            g = ENC_RE.search(ln)
            if g:
                v = [int(x) for x in g.groups()[:-1]]
                enc.append(dict(dl=v[2], dr=v[3], st=v[8], sp=v[9],
                                mode=g.groups()[-1]))
    run = [r for r in enc if r["mode"] == "V" and r["sp"] > 0
           and (r["dl"] + r["dr"]) > 8]
    m["moved_samples"] = len(run)
    if len(run) >= 8:
        xs = [r["st"] for r in run]
        ys = [r["dr"] - r["dl"] for r in run]
        mx, my = st.mean(xs), st.mean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        if den:
            # Am la DUNG: steer duong = re phai = banh phai cham hon.
            m["ticks_per_steer"] = round(
                sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den, 2)
        m["diff_abs_median"] = round(st.median([abs(y) for y in ys]), 1)
    return m


def git_head():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
            stderr=open(os.devnull, "w")).decode().strip()
    except Exception:
        return None


def cmd_save(run_dir, label):
    run_dir = run_dir.rstrip("/")
    if not os.path.isdir(run_dir):
        print("Khong thay thu muc: %s" % run_dir)
        return 1
    dest = os.path.join(BASE_DIR, label)
    if os.path.exists(dest):
        print("Da co baseline ten '%s'. Xoa thu muc do truoc neu muon ghi lai:"
              % label)
        print("   %s" % dest)
        return 1
    os.makedirs(dest)

    # Config: uu tien ban snapshot CUA LAN CHAY DO. Confg.py hien tai co the
    # da doi nhieu lan ke tu luc chay - dung no la ghi sai lich su.
    # 2026-09-11: KHONG lay Confg.py hien tai lam thay the. Lan chay truoc
    # ngay nay khong co snapshot, va Confg.py da doi nhieu lan ke tu do -
    # ghi ban hien tai vao mot baseline lich su la tu tay tao ra mot ban ghi
    # sai, du co canh bao kem theo. Thieu thi ghi la THIEU; so do ket qua
    # van con gia tri ma khong can bo thong so.
    snap = os.path.join(run_dir, "Confg_snapshot.py")
    from_snapshot = os.path.exists(snap)
    if from_snapshot:
        shutil.copy(snap, os.path.join(dest, "Confg.py"))

    # Firmware doc tu file HIEN TAI - khong co cach nao biet firmware nao
    # dang nam trong chip luc chay. Ghi ro trong meta de khong tuong la no
    # duoc chup lai cung lan chay.
    fw = read_consts(FIRMWARE, FW_KEYS)
    with open(os.path.join(dest, "firmware.txt"), "w") as f:
        for k in FW_KEYS:
            f.write("%s = %s\n" % (k, fw.get(k, "?")))

    meta = {
        "label": label,
        "run_dir": run_dir,
        "git_head_when_saved": git_head(),
        "config_from_run_snapshot": from_snapshot,
        "config": (read_consts(os.path.join(dest, "Confg.py"), CFG_KEYS)
                   if from_snapshot else None),
        "firmware": fw,
        "firmware_note": "doc tu main.cpp hien tai luc luu, KHONG phai chup "
                         "cung lan chay",
        "measured": measure(run_dir),
    }
    with open(os.path.join(dest, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print("Da luu baseline '%s' -> %s" % (label, dest))
    if not from_snapshot:
        print("  LUU Y: lan chay nay khong co Confg_snapshot.py (chay truoc")
        print("  2026-09-11) nen KHONG luu duoc bo thong so - chi luu so do")
        print("  ket qua. 'restore' se khong dung duoc voi baseline nay.")
    show(meta)
    return 0


def show(meta):
    mm = meta.get("measured", {})
    print("  lan chay : %s" % meta.get("run_dir"))
    if meta.get("config") is None:
        print("  thong so : KHONG CO (lan chay khong luu snapshot)")
    print("  ket qua  : %s khung / %ss, %s fps, LINE %s%%"
          % (mm.get("frames", "?"), mm.get("duration_s", "?"),
             mm.get("fps", "?"), mm.get("line_pct", "?")))
    print("             |err| trung vi %s (p90 %s), lech phai %s"
          % (mm.get("err_abs_median", "?"), mm.get("err_abs_p90", "?"),
             mm.get("err_right_frac", "?")))
    if mm.get("moved_samples"):
        print("             xe CHAY: %s mau, chenh lech %s tick/steer, "
              "|chenh lech| trung vi %s"
              % (mm["moved_samples"], mm.get("ticks_per_steer", "?"),
                 mm.get("diff_abs_median", "?")))
    else:
        print("             xe KHONG CHAY trong lan nay - so do dieu khien")
        print("             va co khi khong co gia tri")


def cmd_list():
    if not os.path.isdir(BASE_DIR):
        print("Chua co baseline nao. Luu bang:")
        print("   python3 baseline.py save runs_pulled/<lan_chay> \"<nhan>\"")
        return 0
    names = sorted(os.listdir(BASE_DIR))
    if not names:
        print("Chua co baseline nao.")
        return 0
    for n in names:
        mp = os.path.join(BASE_DIR, n, "meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        print("\n=== %s ===" % n)
        show(meta)
    print()
    return 0


def cmd_diff(label):
    mp = os.path.join(BASE_DIR, label, "meta.json")
    if not os.path.exists(mp):
        print("Khong thay baseline '%s'" % label)
        return 1
    meta = json.load(open(mp))
    if meta.get("config") is None:
        print("Baseline '%s' khong luu duoc bo thong so, khong so sanh duoc."
              % label)
        print("No chi con gia tri lam moc KET QUA:")
        show(meta)
        return 1
    now_cfg = read_consts(os.path.join(HERE, "Confg.py"), CFG_KEYS)
    now_fw = read_consts(FIRMWARE, FW_KEYS)
    for title, old, new in (("CONFIG", meta.get("config", {}), now_cfg),
                            ("FIRMWARE", meta.get("firmware", {}), now_fw)):
        print("\n--- %s: baseline '%s'  ->  hien tai ---" % (title, label))
        same = True
        for k in sorted(set(list(old) + list(new))):
            a, b = old.get(k, "?"), new.get(k, "?")
            if a != b:
                print("  %-28s %8s  ->  %8s" % (k, a, b))
                same = False
        if same:
            print("  (khong khac gi)")
    print()
    return 0


def cmd_restore(label):
    src = os.path.join(BASE_DIR, label, "Confg.py")
    if not os.path.exists(src):
        print("Khong thay baseline '%s'" % label)
        return 1
    cmd_diff(label)
    print("Se GHI DE Confg.py hien tai bang ban cua baseline '%s'." % label)
    print("Cac hang so FIRMWARE o tren KHONG duoc khoi phuc tu dong - firmware")
    print("phai nap lai bang PlatformIO neu chung khac nhau.")
    ans = input("Ghi de Confg.py? [go 'yes' de dong y] ").strip()
    if ans != "yes":
        print("Bo qua, khong sua gi.")
        return 0
    shutil.copy(os.path.join(HERE, "Confg.py"),
                os.path.join(HERE, "Confg.py.before_restore"))
    shutil.copy(src, os.path.join(HERE, "Confg.py"))
    print("Da ghi de. Ban cu luu o Confg.py.before_restore")
    print("Nho deploy len Jetson roi moi chay.")
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    c = argv[1]
    if c == "list":
        return cmd_list()
    if c == "save" and len(argv) >= 4:
        return cmd_save(argv[2], argv[3])
    if c == "diff" and len(argv) >= 3:
        return cmd_diff(argv[2])
    if c == "restore" and len(argv) >= 3:
        return cmd_restore(argv[2])
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
