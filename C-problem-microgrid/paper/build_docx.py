"""paper_raw.docx -> 论文_C题.docx
后处理：① 目录字段 ② 页眉页脚（页码域）③ 三线表 ④ 图片限宽 ⑤ 中文字体与正文排版
"""
from docx import Document
from docx.shared import Pt, Inches, Emu, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

SRC = "paper_raw.docx"
DST = "论文_C题_微电网购电策略.docx"

doc = Document(SRC)

# ---------------- 工具函数 ----------------


def set_eastasia(style, ea="宋体", latin="Times New Roman"):
    style.font.name = latin
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    rf.set(qn("w:eastAsia"), ea)
    rf.set(qn("w:ascii"), latin)
    rf.set(qn("w:hAnsi"), latin)


def add_field(paragraph, instr):
    """插入 Word 域（如 TOC / PAGE）"""
    r = paragraph.add_run()
    fld = OxmlElement("w:fldChar")
    fld.set(qn("w:fldCharType"), "begin")
    r._r.append(fld)
    r2 = paragraph.add_run()
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = instr
    r2._r.append(it)
    r3 = paragraph.add_run()
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "separate")
    r3._r.append(fld2)
    r4 = paragraph.add_run("（请在 Word 中按 Ctrl+A → F9 更新目录/页码）")
    r4.font.size = Pt(9)
    r5 = paragraph.add_run()
    fld3 = OxmlElement("w:fldChar")
    fld3.set(qn("w:fldCharType"), "end")
    r5._r.append(fld3)
    return paragraph


def make_three_line(table):
    """三线表：顶线/底线 1.5pt，表头下线 0.75pt，其余无线"""
    tbl = table._tbl
    tblPr = tbl.tblPr
    # 清除现有边框定义
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for tag, sz in (("top", "12"), ("bottom", "12")):
        el = OxmlElement(f"w:{tag}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), sz)  # 单位 1/8 pt
        el.set(qn("w:color"), "000000")
        borders.append(el)
    for tag in ("left", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{tag}")
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        borders.append(el)
    tblPr.append(borders)
    # 表头行下线
    if len(table.rows):
        for cell in table.rows[0].cells:
            tcPr = cell._tc.get_or_add_tcPr()
            for old in tcPr.findall(qn("w:tcBorders")):
                tcPr.remove(old)
            tb = OxmlElement("w:tcBorders")
            el = OxmlElement("w:bottom")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "6")
            el.set(qn("w:color"), "000000")
            tb.append(el)
            tcPr.append(tb)
        # 表头加粗
        for cell in table.rows[0].cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.bold = True
    # 表内字号与行距
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                for r in p.runs:
                    r.font.size = Pt(9)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER


# ---------------- 1. 全局样式 ----------------
normal = doc.styles["Normal"]
set_eastasia(normal, "宋体")
normal.font.size = Pt(12)
normal.paragraph_format.line_spacing = 1.3
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

for hname, ea, size in (("Heading 1", "黑体", 15), ("Heading 2", "黑体", 13),
                        ("Heading 3", "黑体", 12), ("Title", "黑体", 17)):
    try:
        st = doc.styles[hname]
        set_eastasia(st, ea)
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.font.bold = True
    except KeyError:
        pass

# ---------------- 2. 图片限宽并居中 ----------------
MAXW = Inches(6.1)
for shp in doc.inline_shapes:
    if shp.width > MAXW:
        ratio = MAXW / shp.width
        shp.height = Emu(int(shp.height * ratio))
        shp.width = Emu(int(MAXW))

# ---------------- 3. 三线表 ----------------
for t in doc.tables:
    make_three_line(t)

# ---------------- 4. 页眉页脚 ----------------
sec = doc.sections[0]
sec.different_first_page_header_footer = False

hp = sec.header.paragraphs[0]
hp.text = "2026 年全国大学生数学建模竞赛 C 题论文"
hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for r in hp.runs:
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    rpr = r._r.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    rf.set(qn("w:eastAsia"), "宋体")
# 页眉下线
pPr = hp._p.get_or_add_pPr()
pbdr = OxmlElement("w:pBdr")
btm = OxmlElement("w:bottom")
btm.set(qn("w:val"), "single")
btm.set(qn("w:sz"), "4")
btm.set(qn("w:color"), "999999")
pbdr.append(btm)
pPr.append(pbdr)

fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r1 = fp.add_run("第 ")
add_field(fp, "PAGE")
r2 = fp.add_run(" 页　共 ")
add_field(fp, "NUMPAGES")
r3 = fp.add_run(" 页")
for r in fp.runs:
    r.font.size = Pt(9)

# ---------------- 5. 目录（插在关键词段之后） ----------------
kw_idx = None
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith("关键词"):
        kw_idx = i
        break
if kw_idx is not None:
    anchor = doc.paragraphs[kw_idx]
    # 在关键词段后插入：分页符 + “目 录” + TOC 域 + 分页符
    p_break = OxmlElement("w:p")
    r = OxmlElement("w:r")
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    r.append(br)
    p_break.append(r)
    anchor._p.addnext(p_break)

    p_toc_title = OxmlElement("w:p")
    r1 = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    b = OxmlElement("w:b")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "32")
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:eastAsia"), "黑体")
    rPr.append(rFonts)
    rPr.append(b)
    rPr.append(sz)
    r1.append(rPr)
    t = OxmlElement("w:t")
    t.text = "目　录"
    r1.append(t)
    # 居中
    pPr = OxmlElement("w:pPr")
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center")
    pPr.append(jc)
    p_toc_title.append(pPr)
    p_toc_title.append(r1)
    p_break.addnext(p_toc_title)

    p_toc = OxmlElement("w:p")
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), ' TOC \\o "1-3" \\h \\z \\u ')
    hint = OxmlElement("w:r")
    ht = OxmlElement("w:t")
    ht.text = "（目录将在此生成：请在 Word 中全选后按 F9 更新域）"
    hint.append(ht)
    fld.append(hint)
    p_toc.append(fld)
    p_toc_title.addnext(p_toc)

    p_break2 = OxmlElement("w:p")
    r2 = OxmlElement("w:r")
    br2 = OxmlElement("w:br")
    br2.set(qn("w:type"), "page")
    r2.append(br2)
    p_break2.append(r2)
    p_toc.addnext(p_break2)

doc.save(DST)
print(f"[OK] 已生成 {DST}")
