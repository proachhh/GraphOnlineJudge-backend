import io
from PIL import Image


def compress_image(uploaded_file, max_size=(800, 800), quality=80):
    """
    压缩上传的图片，大幅降低体积。
    - max_size: 最大尺寸，等比缩放
    - quality: JPEG 质量 1-100，PNG 转 JPEG
    - 返回 (bytes, suffix) — 压缩后的图片数据和扩展名
    """
    img = Image.open(uploaded_file)
    suffix = (img.format or 'JPEG').lower()
    if suffix == 'jpeg':
        suffix = 'jpg'

    # 修正方向 (EXIF)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # 转 RGB（JPEG 不支持 RGBA）
    if img.mode in ('RGBA', 'P', 'LA'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background

    # 等比缩放
    img.thumbnail(max_size, Image.LANCZOS)

    # 统一输出为 JPEG（体积小）
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality, optimize=True)
    return buf.getvalue(), 'jpg'
