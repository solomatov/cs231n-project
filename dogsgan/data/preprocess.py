from lxml import etree
from PIL import Image

from dogsgan.data.common import image_dir, annotation_dir, preprocessed_dir, image_size


def annotated_images():
    for image in image_dir.glob('**/*.jpg'):
        name = image.stem
        cls = image.parent.stem

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

        yield image, cls, (xmin, ymin, xmax, ymax)


if __name__ == '__main__':
    preprocessed_dir.mkdir(exist_ok=True)

    n = 0
    for i, c, bounds in annotated_images():
        if n % 1000 == 0:
            print(n)

        xmin, ymin, xmax, ymax = bounds
        im = Image.open(i)

        min_dim = min(xmax - xmin, ymax - ymin)
        crop = im.crop((xmin, ymin, xmin + min_dim, ymin + min_dim))

        result = crop.resize(image_size, Image.ANTIALIAS)
        result = result.convert("RGB")

        (preprocessed_dir / c).mkdir(exist_ok=True)
        target = preprocessed_dir / c / f'{n:06}.jpeg'
        result.save(str(target), quality=95)

        n += 1

