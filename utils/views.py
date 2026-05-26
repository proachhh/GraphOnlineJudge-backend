import os
from django.conf import settings
from account.serializers import ImageUploadForm, FileUploadForm
from utils.shortcuts import rand_str
from utils.api import CSRFExemptAPIView
import logging

logger = logging.getLogger(__name__)


class SimditorImageUploadAPIView(CSRFExemptAPIView):
    request_parsers = ()

    def post(self, request):
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data["image"]
        else:
            return self.response({
                "success": False,
                "msg": "Upload failed",
                "file_path": ""})

        suffix = os.path.splitext(img.name)[-1].lower()
        if suffix not in [".gif", ".jpg", ".jpeg", ".bmp", ".png", ".webp"]:
            return self.response({
                "success": False,
                "msg": "Unsupported file format",
                "file_path": ""})

        # GIF 保留原样，其他格式压缩转 JPEG
        if suffix == ".gif":
            img_name = rand_str(10) + ".gif"
            try:
                with open(os.path.join(settings.UPLOAD_DIR, img_name), "wb") as f:
                    for chunk in img:
                        f.write(chunk)
            except IOError as e:
                logger.error(e)
                return self.response({
                    "success": False,
                    "msg": "Upload Error",
                    "file_path": ""})
        else:
            from utils.image import compress_image
            try:
                data, out_suffix = compress_image(img, max_size=(1200, 1200), quality=85)
                img_name = rand_str(10) + "." + out_suffix
                with open(os.path.join(settings.UPLOAD_DIR, img_name), "wb") as f:
                    f.write(data)
            except Exception as e:
                logger.error(e)
                return self.response({
                    "success": False,
                    "msg": "Image processing failed",
                    "file_path": ""})

        return self.response({
            "success": True,
            "msg": "Success",
            "file_path": f"{settings.UPLOAD_PREFIX}/{img_name}"})


class SimditorFileUploadAPIView(CSRFExemptAPIView):
    request_parsers = ()

    def post(self, request):
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.cleaned_data["file"]
        else:
            return self.response({
                "success": False,
                "msg": "Upload failed"
            })

        suffix = os.path.splitext(file.name)[-1].lower()
        file_name = rand_str(10) + suffix
        try:
            with open(os.path.join(settings.UPLOAD_DIR, file_name), "wb") as f:
                for chunk in file:
                    f.write(chunk)
        except IOError as e:
            logger.error(e)
            return self.response({
                "success": False,
                "msg": "Upload Error"})
        return self.response({
            "success": True,
            "msg": "Success",
            "file_path": f"{settings.UPLOAD_PREFIX}/{file_name}",
            "file_name": file.name})
