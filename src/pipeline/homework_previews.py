"""Keep later board previews within the encrypted diagnostic export budget."""
import base64
import io


def retain_preview(previews, image, row, *, budget=1400000, maximum=6):
    if not image or row['source'] != 'video_frame':
        return
    from PIL import Image
    with Image.open(io.BytesIO(image)) as source:
        source.load(); source = source.convert('RGB'); size = list(source.size)
        encoded = ''
        for quality in (85, 70, 55):
            buf = io.BytesIO(); source.save(buf, format='JPEG', quality=quality)
            encoded = base64.b64encode(buf.getvalue()).decode()
            if len(encoded) <= budget/maximum:
                break
    if len(encoded) > budget:
        return
    previews.append({'seconds': row['seconds'], 'source': row['source'],
                     'original_size': size, 'jpeg_base64': encoded, 'jpeg_quality': quality})
    previews.sort(key=lambda preview: preview['seconds'])
    # Late, more complete board states take priority over early preparation.
    while len(previews) > maximum or sum(len(p['jpeg_base64']) for p in previews) > budget:
        previews.pop(0)
