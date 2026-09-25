"""把 tables.json 转成 LaTeX 表格片段（booktabs 三线表），供 main.tex \\input"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
o = json.loads((HERE / "tables.json").read_text(encoding="utf-8"))
meta = o["_meta"]
T1 = meta["t1_slots"]
T2 = meta["t2_slots"]
DATES = {"0320": "3 月 20 日", "0621": "6 月 21 日",
         "0923": "9 月 23 日", "1221": "12 月 21 日"}
KEYS = ["0320", "0621", "0923", "1221"]


def f(v, nd=2):
    return f"{v:,.{nd}f}"


L = []


def add(s=""):
    L.append(s)


# ---------------- 问题1 ----------------
p1 = o["P1"]
add(r"% ================= 问题1 表1/表2")
add(r"\begin{table}[H]\centering\small")
add(r"\caption{问题1：微网在指定时间段的购电量及全天购电量和购电费}\label{tab:p1t1}")
add(r"\begin{tabular}{lrr}")
add(r"\toprule")
add(r"项目 & 数值 & 单位\\")
add(r"\midrule")
for s, v in zip(T1, p1["t1"]):
    add(f"{s} & {f(v,3)} & kWh\\\\")
add(r"\midrule")
add(f"全天购电量 & {f(p1['day_purch'],3)} & kWh\\\\")
add(f"全天购电费 & {f(p1['day_fee'],3)} & 元\\\\")
add(r"\bottomrule\end{tabular}\end{table}")

add(r"\begin{table}[H]\centering\small")
add(r"\caption{问题1：储能设备在指定时间段的充放电量及储电量}\label{tab:p1t2}")
add(r"\begin{tabular}{lrr}")
add(r"\toprule")
add(r"时间段 & 充电量 (kWh) & 放电量 (kWh)\\")
add(r"\midrule")
for s, c, dd in zip(T2, p1["t2_chg"], p1["t2_dis"]):
    add(f"{s} & {f(c,1)} & {f(dd,1)}\\\\")
add(r"\midrule")
add(f"0:00 储电量 & \\multicolumn{{2}}{{c}}{{{p1['E0']:,.0f} kWh}}\\\\")
add(f"24:00 储电量 & \\multicolumn{{2}}{{c}}{{{p1['E24']:,.1f} kWh}}\\\\")
add(r"\bottomrule\end{tabular}\end{table}")

# ---------------- 问题2/3/4-2/4-3 通用三组表 ----------------


def three_tables(tag, label, cap, show_adj=False):
    d = o[tag]
    ex = d["0320"]
    # 表1 购电量
    add(r"% ================= " + tag)
    add(r"\begin{table}[H]\centering\small")
    add(rf"\caption{{{cap}：微网在指定日期指定时段的购电量及全天购电量与购电费}}\label{{tab:{label}t1}}")
    cols = "l" + "r" * (len(T1) + 2 + (1 if show_adj else 0))
    add("\\resizebox{\\textwidth}{!}{\\begin{tabular}{" + cols + "}")
    add(r"\toprule")
    head = "日期 & " + " & ".join(T1) + " & 全天购电量 & 全天购电费"
    if show_adj:
        head += " & 紧急购电量"
    add(head + r"\\")
    add(r"\midrule")
    for k in KEYS:
        r = d[k]
        row = DATES[k] + " & " + " & ".join(f(v) for v in r["t1"]) \
            + f" & {f(r['day_purch'],1)} & {f(r['day_fee'],1)}"
        if show_adj:
            row += f" & {f(r['emg'],1)}"
        add(row + r"\\")
    add(r"\bottomrule\end{tabular}}\end{table}")
    # 表2 储能
    add(r"\begin{table}[H]\centering\small")
    add(rf"\caption{{{cap}：储能设备在指定日期指定时间段的充放电量及储电量}}\label{{tab:{label}t2}}")
    add("\\resizebox{\\textwidth}{!}{\\begin{tabular}{l" + "r" * (len(T2) * 2) + "r}")
    add(r"\toprule")
    add("日期 & " + " & ".join(T2) + r" & \\")
    add(r" & " + " & ".join([r"\multicolumn{1}{c}{充}"] * len(T2)) + " & \\multicolumn{1}{c}{0:00 储电量}\\\\")
    add(r" & " + " & ".join([r"\multicolumn{1}{c}{放}"] * len(T2)) + " & \\multicolumn{1}{c}{24:00 储电量}\\\\")
    add(r"\midrule")
    for k in KEYS:
        r = d[k]
        add(DATES[k] + " & " + " & ".join(f(c, 1) for c in r["t2_chg"])
            + f" & {r['E0']:,.0f}" + r"\\")
        add(" & " + " & ".join(f(c, 1) for c in r["t2_dis"])
            + f" & {r['E24']:,.0f}" + r"\\")
        add(r"\midrule")
    L.pop()  # 去掉最后一条多余 midrule
    add(r"\bottomrule\end{tabular}}\end{table}")


# 表3 紧急购电量（题面指定表）
def emergency_table(tag3, tag43):
    d3, d43 = o[tag3], o[tag43]
    add(r"% ================= 表3 紧急购电量")
    add(r"\begin{table}[H]\centering\small")
    add(r"\caption{微网在指定日期的紧急购电量（kWh）}\label{tab:p4t3}")
    add(r"\begin{tabular}{lrr}")
    add(r"\toprule")
    add(r"日期 & 问题3（固定电价） & 问题4（波动电价）\\")
    add(r"\midrule")
    for k in KEYS:
        add(f"{DATES[k]} & {f(d3[k]['emg'],2)} & {f(d43[k]['emg'],2)}\\\\")
    add(r"\bottomrule\end{tabular}\end{table}")


three_tables("P2", "p2", "问题2", show_adj=False)
three_tables("P3", "p3", "问题3", show_adj=True)
three_tables("P42", "p42", "问题4-2（完美信息）", show_adj=False)
three_tables("P43", "p43", "问题4-3（滚动调整）", show_adj=True)
emergency_table("P3", "P43")

# ---- 按注释标记行拆分（PDF 版：resizebox 防溢出） ----
def _block(startmark, endmarks):
    i = next(k for k, s in enumerate(L) if s.strip() == startmark)
    j = len(L)
    for m in endmarks:
        k = next((q for q, s in enumerate(L) if s.strip() == m), None)
        if k is not None:
            j = min(j, k)
    return L[i:j]


(HERE / "tables_gen_p1.tex").write_text("\n".join(_block("% ================= 问题1 表1/表2", ["% ================= P2"])), encoding="utf-8")
(HERE / "tables_gen_p2.tex").write_text("\n".join(_block("% ================= P2", ["% ================= P3"])), encoding="utf-8")
(HERE / "tables_gen_p3.tex").write_text("\n".join(_block("% ================= P3", ["% ================= P42"])), encoding="utf-8")
(HERE / "tables_gen_p4.tex").write_text("\n".join(L[next(k for k, s in enumerate(L) if s.strip() == "% ================= P42"):]), encoding="utf-8")

# ---- plain 版（供 pandoc 转 Word：\resizebox 会被 pandoc 丢弃，须用裸 tabular） ----
LP = [s.replace("\\resizebox{\\textwidth}{!}{\\begin{tabular}", "\\begin{tabular}")
        .replace("\\bottomrule\\end{tabular}}", "\\bottomrule\\end{tabular}") for s in L]
k1 = next(k for k, s in enumerate(LP) if s.strip() == "% ================= 问题1 表1/表2")
k2 = next(k for k, s in enumerate(LP) if s.strip() == "% ================= P2")
k3 = next(k for k, s in enumerate(LP) if s.strip() == "% ================= P3")
k4 = next(k for k, s in enumerate(LP) if s.strip() == "% ================= P42")
(HERE / "tables_plain_p1.tex").write_text("\n".join(LP[:k2]), encoding="utf-8")
k2 = next(k for k, s in enumerate(LP) if s.strip() == "% ================= P2")
k3 = next(k for k, s in enumerate(LP) if s.strip() == "% ================= P3")
k4 = next(k for k, s in enumerate(LP) if s.strip() == "% ================= P42")
(HERE / "tables_plain_p2.tex").write_text("\n".join(LP[k2:k3]), encoding="utf-8")
(HERE / "tables_plain_p3.tex").write_text("\n".join(LP[k3:k4]), encoding="utf-8")
(HERE / "tables_plain_p4.tex").write_text("\n".join(LP[k4:]), encoding="utf-8")
print("[OK] tables_gen_p*.tex（PDF）与 tables_plain_p*.tex（Word）已写出")
