---
layout : post
title : Поваренная книга обработки картинок
---

Здесь будут собираны рецепты по имитации эффектов различных пунктов меню в фотошопе. Полезно, например, для создания пайплайна предобработки в классическом компьютерном зрении или для аугментации данных в глубоком обучении.

## Select

Часто бывает нужно обработать только часть изображения. Например, заблюрить фон, оставив передний план четким.

![Меню select в GIMP](https://user-images.githubusercontent.com/2237541/143660833-d5015a0b-9db1-4118-94b2-7b8ed9f9787f.png)

### Базовый пример

Нужна маска - изображение того же размера, в котором для каждого пикселя описано, какой части он принадлежит - обрабатываемой, или игнорируемой.  Так изображение с наложенной маской может создать краевые эффекты, обычно имеет смысл накладывать ее уже после обработки. Но можно и до.
Например, если мы хотим обработать только самые темные области изображения.

```python
#конвертируем в gray scale
gray = np.mean(image, axis = 2, keepdims = True)
# маска будет содержать 0 для темных пикселей и 1 для светлых
# конвертируем boolean'ы во флоаты
mask = (gray > 0.1).astype(float)
processed_image = process(image)
result = processed_image*mask + image*(1-mask)
```

### Увеличение-уменьшение размеров (эрозия-дилатация)

Если маска состоит из 0 и 1, можно ее увеличивать и уменьшать, как в пунктах меню Shrink и Grow. Из комбинации можно найти границу:

```python
# уменьшение
erode = scipy.ndimage.binary_erosion(input, iterations = ...)
# увеличение
dilate = scipy.ndimage.binary_erosion(input, iterations = ...)
# граница
border = dilate - erode
```

### Маска с размытием

Резкие края у маски часто нежелательны, и делают ее заметной на результате. Чтобы их не было, маску можно размыть:

```python
mask = (gray > 0.1).astype(float)
smoothed_mask = scipy.ndimage.gaussian_filter(mask, sigma)
```

При этом код `result = processed_image*mask + image*(1-mask)` отработает, как надо.


### Выделение по критерию (например, по цвету)

Если нужен какой-то сложный критерий, то имеет смысл векторными операциями превратить картинку в такую, что более белому цвету соответствует большая вероятность соответствия критерую. Затем применить то же, что и выше: `gray > ...`. Для выделения по цвету, можно взять L2-норму как критерий: `sqrt( (r1-r2)^2 + (g1-g2)^2 + (b1-b2)^2 )`. Далее см. комментарии. в коде:

```python
# [r, g, b] - цвет, близкие величины к которому находим
# мы хотим вычесть одну и ту же константу из каждого пикселя
# для этого воспользуемся broadcasting'ом, поставив 1 там, где
# у картинки длина и ширина
color_for_broadcasting = np.reshape([r, g, b], [1, 1, 3])
# Теперь операция `image - color_for_broadcasting` не будет падать
# с несоответствием размеростей
# Первый этап вычисления критерия
squared_diff = (image - color_for_broadcasting)**2
# Второй этап
# в каждом пикселе color_distance находится степень близости цвета
color_distance = np.sqrt(np.sum(squared_diff, axis=2, keepdims=True))
# теперь можно найти маску
mask = color_distance > THRESHOLD
```

Чтобы найти THRESHOLD, имеет смысл посмотреть на содержимое color_distance с помощщью matplotib.pyplot.imshow

```python
import matplotlib.pyplot as plt
plt.imshow(color_distance)
plt.show()
```

Таким же критерием может быть результат работы нейронки, например, предсказанная дистанция.

## Попиксельные обработки (яркость, контраст, уровни, кривые и гамма)

Все эти обработки берут значение пикселя и применяют к нему какую-то функцию, и записывают результат назад. Каждый пиксель обрабатывается независимо. Таким образом, все, что отличает эти операции, это вид применяемой к пикселю функции. "Кривые" дают самую наглядную визуализацию и являются самым мощным и наглядным вариантом. Остальные обработки создают какую-то свою кривую под капотом, но ее не демонстрируют. Поэтому для окончательных тестов того, а что такого закодить, рекомендую переходить сразу к кривым.

### Яркость и контраст

Простейший случай - применить линейную функцию `k*x + b` к яркости каждого пикселя. C numpy это просто:

```python
new_img = k * img + b
```

Число b отвечает за яркость, k - за kontrast. img - матрица с картинкой.

В результате возможен выход самых ярких пикселей за пределы. Если это не допустимо, все, что ниже 0 и выше 1 превращаем в 0 и 1:

```python
new_img = np.clip(k * img + b, 0, 1)
```

Параметр b точно соответствует яркости в редакторе - 0 означает отсутствие изменений, идем в плюс - увеличиваем яркость, в минус - все тусклее.

С параметром k сложнее, потому что в том виде, как он тут написан, он управляет наклоном функции относительно точки пересечения прямой и оси Y (0; b). Это не совсем интуитивный контраст. Для более интуитивного подхода требуется вращать вокруг точки (0.5, 0.5):

```python
new_img = k*(img - 0.5) + b + 0.5
```

Сам параметр k может быть также нелинейно привязан к крутилке в UI, и как оно сделано в конкретном редакторе - вопрос. 

### Максимизация контраста

Иногда требуется максимизировать контраст, то есть, чтобы самый тусклый пиксель имел значение точно 0, самый яркий точно 1:

```python
min_pixel = np.minimum(img)
max_pixel = np.minimum(img)
max_contrast_img = (img - min_pixel) / (max_pixel - min_pixel)
```

### Уровни

Уровни применяют такую же `k*x+b`, но через UI задаются значения этой функции для самого темного и яркого пикселей. Есть более продвинутая версия, где помимо черного и белого ползунка есть еще серый, midpoint. Здесь уже не прямая, а изогнутая кривая.

### Кривые

С помощью пункта меню "кривые" можно нарисовать функцию непосредственно в UI. Если функция кусочно-линейная, ее можно описать с помощью np.interp. Например, вот так можно сделать что-то вроде изменения контраста:

```python
new_img = numpy.interp(img, [0, 1], [0.3, 0.8])
```

Аппроксимация гамма-коррекции с помощью двух отрезков:

```python
new_img = numpy.interp(img, [0, 0.5, 1], [0, 0.8, 1])
```

Для более гладкой кривой интерполировать сплайнами:

```python
tck = interpolate.splrep([0, 0.5, 1], [0, 0.8, 1], s=0)
xnew = np.linspace(0, 1, 100)
ynew = interpolate.splev(xnew, tck, der=0)
```

### Реверс попиксельного алгоритма

Если накрутили что-то в редакторе и нужно перенести это в скрипт, то для реверса нужно создать png со всеми 255 оттенками серого, обработать его настройкой, и сохранить результат, и путем сравнения оригинального и обработанного файлов реверснуть обработку.

![Файл со всеми оттенками серого](https://user-images.githubusercontent.com/2237541/143663498-1b2e122b-4210-4306-9593-2c18eff5afaf.png)

```python
from PIL import Image
import numpy as np

# строка
img = np.array(range(255), dtype=np.uint8)
# превращаем shape в [1, 255, 1]
img = img[np.newaxis, :, np.newaxis]
# превращаем картинку в цветную, дублируя ось channel
img = np.tile(img, (1, 1, 3))
Image.fromarray(img).save("gradient_sample.png")
```

Обрабатываем gradient_sample.png.

![Скриншот curves из gimp](https://user-images.githubusercontent.com/2237541/143663534-d6353615-6ebf-4f18-8471-1685f81e1d31.png)

После обработки можно вычленить нужную информацию и, для удобства работы с дробными значениями, превратить ее из массива в функцию.

```python
import scipy.interpolate
import matplotlib.pyplot as plt

with Image.open('gradient_sample_out.png') as img:
    img_arr = np.asarray(img)
    # функция конвертации красного
    img_arr_r = np.squeeze(img_arr[:, :, 0])
    xs = np.arange(255)
    ys = img_arr_r
    fun = scipy.interpolate.InterpolatedUnivariateSpline(xs, ys)
    plt.plot(fun(xs))
    plt.show()
```

![Результат реверса](https://user-images.githubusercontent.com/2237541/143663779-e31fa8e9-cee6-40e1-a931-23d6396a5703.png)

## Размытие

Выбор по умолчанию - размытие по Гауссу. Оно довольно неплохо имитирует то, что происходит в расфокусированной оптике.

Параметр sigma управляет величиной размытия. Для совсем ленивых есть уже готовая функция в scipy:

```python
filtered_image = scipy.ndimage.gaussian_filter(input, sigma)
```

Этот прием уже был использован для размытия маски.

Другие виды размытия описаны в https://galchinsky.github.io/2021/11/25/KnowYourOptics.html

### Реверс размытия

Первым делом нужно определить, одинаковое ли размытие везде в картинке, или в разных областях размывается по-разному. Для этого делаем белый файл с черными точками в виде сетки и обрабатываем фильтром.

* Если результирующие размытые точки везде одинаковые - ура. Измеряем величину размытого пятна в пикселях. Затем создаем черный файл такого размера и устанавливаем посередине белую точку:

```python
from PIL import Image
import numpy as np

img = np.zeros((255, 255, 3), dtype=np.uint8)
img[img.shape[0]//2, img.shape[0]//2, :] = 255
Image.fromarray(img).save("delta.png")
```

![Черный квадрат с белой точкой посередине](https://user-images.githubusercontent.com/2237541/143663962-57de4a88-5e6e-4792-9692-154579dad74b.png)

Обрабатываем ее в редакторе. 

![Motion blur в GIMP](https://user-images.githubusercontent.com/2237541/143663988-8947442a-35e0-4fbc-baee-f619896db7a4.png)

Получаенная картинка - это и есть то ядро, которое нужно скормить в convolve2d. Видно, что motion blur - имеет ядро в виде линии.

```python
import scipy.interpolate
import matplotlib.pyplot as plt

with Image.open('delta_out.png') as img:
    img_arr = np.asarray(img)
    img_arr = img_arr.astype(float)/255.0
    # так как в случае motion blur обработка идентичная для всех каналов
    # берем только красный канал
    # помимо этого, выделяем только центральную область для более
    # быстрой свертки
    h, w, c = img_arr.shape
    img_arr = img_arr[h//2-16:h//2+32, w//2-16:w//2+32, :]
    img_arr = img_arr[:, :, 0]
    plt.imshow(img_arr)
    plt.show()
```

![Ядро motion blur](https://user-images.githubusercontent.com/2237541/143664112-1d4adfa5-1f86-428c-bd9d-10cfb2acc5e0.png)

GIMP почему-то создал неотцентрированное ядро, поэтому картинка в итоге съедет. Чтобы она не съезжала, можно применить следующий спосб:

```python
x, y = scipy.ndimage.measurements.center_of_mass(a)
a = scipy.ndimage.interpolation.shift(a, [kernel_size//2 - x, kernel_size//2 - y])
```

Делать это мы, конечно же, не будем. Вот так его можно применить:

```python
import scipy.signal
import scipy.misc
f = scipy.misc.face().copy()
f[:, :, 0] = scipy.signal.convolve2d(f[:, :, 0], img_arr, 'same')
f[:, :, 1] = scipy.signal.convolve2d(f[:, :, 1], img_arr, 'same')
f[:, :, 2] = scipy.signal.convolve2d(f[:, :, 2], img_arr, 'same')
plt.imshow(f)
plt.show()
```

![Оригинальный енот](https://user-images.githubusercontent.com/2237541/143664574-98590c14-b7ef-4e88-a0ea-25d840465a13.png)
![Енот с моушен блюром](https://user-images.githubusercontent.com/2237541/143664359-2ff85146-6284-425b-8dee-aa43517c14ce.png)

Картинка съехала и образовалась черная полоса. Чтобы это замаскировать, если центрировать кернел не вариант, можно указать boundary='symm'.

Если результирующие точки разные - все сложнее, но подход тот же. Нужно скормить гимпу сетку точек, выделить из размытого файла ядра, и дальше интерполировать между разными ядрами при применении свертки.

## Повышение резкости

Размытие убирает высокие частоты, оставляя низкие. А значит, разность между оригинальным изображением и размытием даст, наоборот, только высокие частоты. Если сложить их с оригиналом, получим эффект увеличения резкости. Называется это unsharp mask и применялось еще с пленочной фотографией. Он есть в scikit-image:

```python
sharpened_image = skimage.filters.unsharp_mask(input, radius=3, amount=0.5)
```

Внутри он устроен, как описано выше: размываем, находим разность, складываем  с оригиналом:

```python
filtered_image = scipy.ndimage.gaussian_filter(input, radius)
sharpened_image = input + amount*(input - filtered_image)
```

В фоторедакторах есть еще один этап: там, где изображение малоконтрастно, оставляем оригинал, комбинируя с помощью маски по рецепту из раздела Select. Это позволяет не вытягивать шумы, ведь шум - это что-то с небольшой энергией, создающее высокочастотную рябь.

```python
mask = np.absolute(input - sharpened_image) > threshold
sharpened_with_threshold = input * (1 - mask) + sharpened_image * mask
```

`input` и `sharpened_image`:

![Оригинальный енот](https://user-images.githubusercontent.com/2237541/146315152-f45bf9b3-97c6-4426-abe9-52c73a7d4e60.png) ![Енот с повышенной резкостью](https://user-images.githubusercontent.com/2237541/146320104-0b1ff721-2369-44e1-ba93-57d95cdf66f4.png)

`mask` и `sharpened_with_threshold`:

![Маска резкости](https://user-images.githubusercontent.com/2237541/146315269-3b16006a-2323-4903-9b29-8f0a6a2860ff.png) ![Енот с повышенной резкостью и маской](https://user-images.githubusercontent.com/2237541/146320204-70765fff-49ab-4339-a93f-9f0000c8b2ef.png)

Примеры сгенерированы этим скриптом:

```python
import scipy.signal
import scipy.misc
import imageio
import numpy as np
import matplotlib.pyplot as plt

input = scipy.misc.face().astype(float)/255
input = input[300:600, 300:600, :]
radius = 3
amount = 0.5
threshold = 0.1

filtered_image = scipy.ndimage.gaussian_filter(input, radius)
sharpened_image = input + amount*(input - filtered_image)
mask = np.absolute(input - sharpened_image) > threshold
sharpened_with_threshold = input * (1 - mask) + sharpened_image * mask
combined_image = np.clip(np.concatenate([input,
                                         sharpened_image,
                                         mask,
                                         sharpened_with_threshold], axis=0), 0, 1)

imageio.imwrite('input.png', input)
imageio.imwrite('sharpened_image.png', np.clip(sharpened_image, 0, 1))
imageio.imwrite('mask.png', mask.astype(float))
imageio.imwrite('sharpened_with_threshold.png', np.clip(sharpened_with_threshold, 0, 1))
```


