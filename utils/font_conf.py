from PIL import ImageFont

FONT_PATH = "assets/fonts/LXGWWenKai-Regular.ttf"

class FontConfig:
    def __init__(self, max_height, max_width, text, font_path=FONT_PATH):
        self.font_path = font_path
        self.font_size = self._find_font_size(max_height, max_width, len(text))

    def _find_font_size(self, max_height, max_width, n):
        font_size = 10
        while True:
            font = ImageFont.truetype(self.font_path, font_size)
            bbox = font.getbbox("中")
            h = bbox[3] - bbox[1]
            w = bbox[2] - bbox[0]
            if h * w * n >= max_height * max_width * 0.55:
                break
            font_size += 1
        return font_size -1
    
    @property
    def font(self):
        return ImageFont.truetype(self.font_path, self.font_size)