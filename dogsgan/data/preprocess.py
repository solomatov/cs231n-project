"""
   Copyright 2018 JetBrains, s.r.o
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from lxml import etree
from PIL import Image

from dogsgan.data.common import image_dir, annotation_dir, preprocessed_dir, image_size

RESIZE_THRESHOLD = 1.5


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
    skipped = 0
    for i, c, bounds in annotated_images():
        if n % 1000 == 0:
            print(n)

        n += 1

        xmin, ymin, xmax, ymax = bounds

        w = (xmax - xmin)
        h = (ymax - ymin)

        # if w / h > RESIZE_THRESHOLD or h / w > RESIZE_THRESHOLD:
        #     skipped += 1
        #
        #     if skipped % 1000 == 0:
        #         print(f'skipped {skipped}')
        #     continue

        im = Image.open(i)
        result = im.resize(image_size, Image.ANTIALIAS)
        result = result.convert("RGB")

        (preprocessed_dir / c).mkdir(exist_ok=True)
        target = preprocessed_dir / c / f'{n:06}.jpeg'
        result.save(str(target), quality=95)


    print(f'total skipped {skipped}')