"""
check_pdf_issues.py
Check all PDF figures in tcc/images/ for transparency, non-RGB colorspaces,
and blend modes that can cause rendering artifacts in compiled LaTeX PDFs.
Uses PyMuPDF (fitz).
"""
import fitz
import os
import re
import sys

BASE = "C:/DEV_ENV/source/UFRGS/TCC/gs-erp-3d-classification/tcc/images"


def check_pdf(path: str) -> list[str]:
    """Return a list of issue strings for the given PDF path."""
    issues: list[str] = []
    doc = fitz.open(path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_prefix = f"p{page_num + 1}" if len(doc) > 1 else ""

        def tag(msg):
            return f"[{page_prefix}] {msg}" if page_prefix else msg

        # ------------------------------------------------------------------
        # 1. Images with soft masks (alpha channel)
        # ------------------------------------------------------------------
        for xref in range(1, doc.xref_length()):
            try:
                if doc.xref_is_image(xref):
                    img = doc.extract_image(xref)
                    smask = img.get("smask", 0)
                    cs = img.get("colorspace", 0)
                    if smask > 0:
                        issues.append(tag(f"IMAGE xref={xref} has soft-mask/alpha (smask={smask})"))
                    if cs == 4:
                        issues.append(tag(f"IMAGE xref={xref} uses CMYK colorspace"))
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 2. ExtGState: alpha (ca/CA < 1), non-normal blend modes, SMask
        # ------------------------------------------------------------------
        def check_extgstate_xref(eg_xref: int, context: str):
            for gs_key in doc.xref_get_keys(eg_xref):
                gs_val = doc.xref_get_key(eg_xref, gs_key)
                if gs_val[0] == "xref":
                    gs_sub_xref = int(gs_val[1].split()[0])
                    for attr in ("ca", "CA", "BM", "SMask"):
                        try:
                            av = doc.xref_get_key(gs_sub_xref, attr)
                            if av[0] == "null":
                                continue
                            val = av[1]
                            if attr in ("ca", "CA"):
                                try:
                                    if float(val) < 1.0:
                                        issues.append(
                                            tag(f"ALPHA {attr}={val} in {context}/{gs_key}")
                                        )
                                except ValueError:
                                    pass
                            elif attr == "BM":
                                if val not in ("/Normal", "Normal", "/Compatible", "Compatible"):
                                    issues.append(
                                        tag(f"BLEND MODE {val} in {context}/{gs_key}")
                                    )
                            elif attr == "SMask":
                                if val not in ("/None", "None", "null"):
                                    issues.append(
                                        tag(f"SMask set in {context}/{gs_key}: {val[:60]}")
                                    )
                        except Exception:
                            pass
                elif gs_val[0] == "dict":
                    # Inline dict — parse text
                    d = gs_val[1]
                    for attr, pattern in [("ca", r"/ca\s+([\d.]+)"),
                                          ("CA", r"/CA\s+([\d.]+)"),
                                          ("BM", r"/BM\s+/(\w+)")]:
                        for m in re.finditer(pattern, d):
                            val = m.group(1)
                            if attr in ("ca", "CA"):
                                try:
                                    if float(val) < 1.0:
                                        issues.append(
                                            tag(f"ALPHA {attr}={val} in {context}/{gs_key} (inline)")
                                        )
                                except ValueError:
                                    pass
                            elif attr == "BM":
                                if val not in ("Normal", "Compatible"):
                                    issues.append(
                                        tag(f"BLEND MODE /{val} in {context}/{gs_key} (inline)")
                                    )

        try:
            r_val = doc.xref_get_key(page.xref, "Resources")
            if r_val[0] == "xref":
                res_xref = int(r_val[1].split()[0])

                eg_val = doc.xref_get_key(res_xref, "ExtGState")
                if eg_val[0] == "xref":
                    check_extgstate_xref(int(eg_val[1].split()[0]), "ExtGState")
                elif eg_val[0] == "dict":
                    # Parse inline ExtGState dict
                    d = eg_val[1]
                    for attr, pattern in [("ca", r"/ca\s+([\d.]+)"),
                                          ("CA", r"/CA\s+([\d.]+)"),
                                          ("BM", r"/BM\s+/(\w+)")]:
                        for m in re.finditer(pattern, d):
                            val = m.group(1)
                            if attr in ("ca", "CA"):
                                try:
                                    if float(val) < 1.0:
                                        issues.append(tag(f"ALPHA {attr}={val} in ExtGState (page-inline)"))
                                except ValueError:
                                    pass
                            elif attr == "BM":
                                if val not in ("Normal", "Compatible"):
                                    issues.append(tag(f"BLEND MODE /{val} in ExtGState (page-inline)"))
        except Exception:
            pass

        # ------------------------------------------------------------------
        # 3. Form XObjects with transparency groups
        # ------------------------------------------------------------------
        try:
            r_val = doc.xref_get_key(page.xref, "Resources")
            if r_val[0] == "xref":
                res_xref = int(r_val[1].split()[0])
                xobj_val = doc.xref_get_key(res_xref, "XObject")
                xobj_xref_list = []
                if xobj_val[0] == "xref":
                    xobj_xref_list.append(int(xobj_val[1].split()[0]))

                for xobj_xref in xobj_xref_list:
                    for xo_key in doc.xref_get_keys(xobj_xref):
                        xo_val = doc.xref_get_key(xobj_xref, xo_key)
                        if xo_val[0] == "xref":
                            xo_xref = int(xo_val[1].split()[0])
                            try:
                                grp = doc.xref_get_key(xo_xref, "Group")
                                if grp[0] != "null":
                                    issues.append(
                                        tag(f"TRANSPARENCY GROUP in XObject/{xo_key}: {grp[1][:80]}")
                                    )
                            except Exception:
                                pass
                            # Recurse into XObject's own ExtGState
                            try:
                                xo_eg = doc.xref_get_key(xo_xref, "Resources/ExtGState")
                                if xo_eg[0] == "xref":
                                    check_extgstate_xref(
                                        int(xo_eg[1].split()[0]),
                                        f"XObject/{xo_key}/ExtGState",
                                    )
                            except Exception:
                                pass
        except Exception:
            pass

        # ------------------------------------------------------------------
        # 4. Content stream scan for blend mode operators
        # ------------------------------------------------------------------
        try:
            content = page.read_contents().decode("latin-1", errors="replace")
            blend_modes = [
                "Multiply", "Screen", "Overlay", "Darken", "Lighten",
                "ColorDodge", "ColorBurn", "HardLight", "SoftLight",
                "Difference", "Exclusion", "Hue", "Saturation", "Color",
                "Luminosity",
            ]
            for bm in blend_modes:
                if f"/{bm}" in content:
                    issues.append(tag(f"BLEND MODE /{bm} referenced in page content stream"))
        except Exception:
            pass

    doc.close()
    return issues


def main():
    all_issues: dict[str, list[str]] = {}
    pdf_count = 0

    for root, _dirs, files in os.walk(BASE):
        for f in sorted(files):
            if not f.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, BASE)
            pdf_count += 1
            size_kb = os.path.getsize(path) // 1024
            try:
                issues = check_pdf(path)
                status = "ISSUES" if issues else "OK"
                print(f"[{status}] {rel}  ({size_kb} KB)")
                for iss in issues:
                    print(f"         -> {iss}")
                if issues:
                    all_issues[rel] = issues
            except Exception as e:
                print(f"[ERROR]  {rel}: {e}")
                all_issues[rel] = [f"Exception: {e}"]

    print()
    print("=" * 72)
    print(f"Checked {pdf_count} PDF file(s).")
    if all_issues:
        print(f"FILES WITH POTENTIAL RENDERING ISSUES ({len(all_issues)}):")
        for rel, issues in all_issues.items():
            print(f"  {rel}")
            for iss in issues:
                print(f"    - {iss}")
    else:
        print("No rendering issues found in any PDF.")


if __name__ == "__main__":
    main()
