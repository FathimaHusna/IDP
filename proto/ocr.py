from __future__ import annotations

from pathlib import Path
from typing import Optional
import io
import os
import shutil


def _avg_conf_from_tesseract_data(data: dict) -> Optional[float]:
    try:
        confs = data.get("conf", [])
        nums = [float(c) for c in confs if str(c).strip() not in {"-1", ""}]
        if not nums:
            return None
        # Normalize from 0-100 to 0-1
        return max(0.0, min(1.0, sum(nums) / (100.0 * len(nums))))
    except Exception:
        return None


def _prepare_image(img):
    try:
        from PIL import ImageOps, ImageFilter

        # Normalize orientation and convert to grayscale
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")
        # Improve contrast and sharpness
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)
        # Upscale small images for better OCR
        w, h = img.size
        if max(w, h) < 1000:
            scale = 2
            img = img.resize((w * scale, h * scale))
        return img
    except Exception:
        return img


def _ocr_pil_image(img) -> dict:
    try:
        import pytesseract  # type: ignore
        from pytesseract import Output  # type: ignore

        img = _prepare_image(img)
        # Try English; if Arabic is installed, include it to improve bilingual docs
        config = "--oem 3 --psm 6 -l eng"
        try:
            from pytesseract import get_languages

            langs = get_languages(config="") or []
            if any(l.startswith("ara") for l in langs):
                config = "--oem 3 --psm 6 -l eng+ara"
        except Exception:
            pass
        text = pytesseract.image_to_string(img, config=config)
        data = pytesseract.image_to_data(img, output_type=Output.DICT, config=config)
        q = _avg_conf_from_tesseract_data(data) or 0.7
        return {"text": text or "", "quality": q}
    except Exception:
        # pytesseract or engine not available
        return {"text": "", "quality": 0.5}


def ocr_from_bytes(data: bytes, suffix: Optional[str] = None) -> dict:
    suf = (suffix or "").lower().lstrip(".")
    # Treat text-like as UTF-8
    if suf in {"txt", "text"}:
        try:
            return {"text": data.decode("utf-8", errors="ignore"), "quality": 0.98}
        except Exception:
            return {"text": "", "quality": 0.5}

    if suf in {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}:
        try:
            from PIL import Image  # type: ignore
            import io

            img = Image.open(io.BytesIO(data))
            return _ocr_pil_image(img)
        except Exception:
            return {"text": "", "quality": 0.5}

    if suf == "pdf":
        # Fast path: extract embedded text without OCR
        try:
            try:
                from pypdf import PdfReader  # type: ignore
            except Exception:
                PdfReader = None  # type: ignore
            if PdfReader is not None:
                reader = PdfReader(io.BytesIO(data))
                # Limit pages for speed
                max_pages = min(len(reader.pages), 5)
                texts = []
                for i in range(max_pages):
                    try:
                        t = reader.pages[i].extract_text() or ""
                        texts.append(t)
                    except Exception:
                        break
                raw_text = "\n".join(texts).strip()
                # If we found any embedded text, skip OCR for speed/robustness
                if len(raw_text) > 0:
                    return {"text": raw_text, "quality": 0.95}
        except Exception:
            pass
        # OCR path: limit pages and DPI to avoid timeouts/latency on large PDFs
        try:
            # Skip OCR entirely if Poppler not installed
            if shutil.which("pdftoppm") is None:
                return {"text": "", "quality": 0.5}
            from pdf2image import convert_from_bytes  # type: ignore

            # Heuristic: if the file is large, only OCR first page at lower DPI
            data_size_mb = len(data) / (1024 * 1024)
            fast_env = os.environ.get("IDP_OCR_FAST", "0") == "1"
            last_page = 1 if (data_size_mb > 6 or fast_env) else 3
            dpi = 200 if (data_size_mb > 6 or fast_env) else 300
            pages = convert_from_bytes(data, fmt="png", first_page=1, last_page=last_page, dpi=dpi)
            texts = []
            qualities = []
            for pg in pages:
                r = _ocr_pil_image(pg)
                texts.append(r.get("text", ""))
                qualities.append(r.get("quality", 0.5))
            text = "\n".join(texts).strip()
            q = sum(qualities) / len(qualities) if qualities else 0.6
            return {"text": text, "quality": q}
        except Exception:
            return {"text": "", "quality": 0.5}

    # Unknown: best-effort UTF-8 decode
    try:
        return {"text": data.decode("utf-8", errors="ignore"), "quality": 0.6}
    except Exception:
        return {"text": "", "quality": 0.5}


def ocr_text(path: str) -> dict:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".txt":
        txt = p.read_text(encoding="utf-8", errors="ignore")
        return {"text": txt, "quality": 0.98}
    if suf in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        try:
            from PIL import Image  # type: ignore

            img = Image.open(p)
            return _ocr_pil_image(img)
        except Exception:
            return {"text": "", "quality": 0.5}
    if suf == ".pdf":
        # Attempt fast text extraction first
        try:
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(str(p))
            max_pages = min(len(reader.pages), 5)
            texts = []
            for i in range(max_pages):
                try:
                    texts.append(reader.pages[i].extract_text() or "")
                except Exception:
                    break
            raw_text = "\n".join(texts).strip()
            if len(raw_text) > 0:
                return {"text": raw_text, "quality": 0.95}
        except Exception:
            pass
        # OCR fallback with limited pages/DPI
        try:
            # Skip OCR if Poppler is not present
            if shutil.which("pdftoppm") is None:
                return {"text": "", "quality": 0.5}
            from pdf2image import convert_from_path  # type: ignore
            # Heuristic based on file size
            size_mb = p.stat().st_size / (1024 * 1024)
            fast_env = os.environ.get("IDP_OCR_FAST", "0") == "1"
            last_page = 1 if (size_mb > 6 or fast_env) else 3
            dpi = 200 if (size_mb > 6 or fast_env) else 300
            pages = convert_from_path(str(p), fmt="png", first_page=1, last_page=last_page, dpi=dpi)
            texts = []
            qualities = []
            for pg in pages:
                r = _ocr_pil_image(pg)
                texts.append(r.get("text", ""))
                qualities.append(r.get("quality", 0.5))
            text = "\n".join(texts).strip()
            q = sum(qualities) / len(qualities) if qualities else 0.6
            return {"text": text, "quality": q}
        except Exception:
            return {"text": "", "quality": 0.5}

    # Fallback
    return {"text": "", "quality": 0.5}
