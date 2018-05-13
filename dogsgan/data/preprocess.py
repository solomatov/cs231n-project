from pathlib import Path
from lxml import etree
from PIL import Image

data_dir = Path('data')
image_dir = data_dir / 'Images'
annotation_dir = data_dir / 'Annotation'
preprocessed_dir = data_dir / 'Preprocessed'

target_size = (128, 128)


def annotated_images():
    for image in image_dir.glob('**/*.jpg'):
        name = image.stem
        parent_relative = image.parent.relative_to(image_dir)
        annotation = annotation_dir / parent_relative / name

        with annotation.open() as file:
            tree = etree.parse(file)

            def get_int_attr(attr):
                return int(tree.findall(f'.//{attr}')[0].text)

            xmin = get_int_attr('xmin')
            ymin = get_int_attr('ymin')
            xmax = get_int_attr('xmax')
            ymax = get_int_attr('ymax')

        yield image, (xmin, ymin, xmax, ymax)


if __name__ == '__main__':
    preprocessed_dir.mkdir(exist_ok=True)

    n = 0
    for i, bounds in annotated_images():
        if n % 1000 == 0:
            print(n)

        xmin, ymin, xmax, ymax = bounds
        im = Image.open(i)

        min_dim = min(xmax - xmin, ymax - ymin)
        crop = im.crop((xmin, ymin, xmin + min_dim, ymin + min_dim))

        result = crop.resize(target_size)
        result = result.convert("RGB")

        target = preprocessed_dir / f'{n:06}.jpeg'
        result.save(str(target), quality=95)

        n += 1

