# Общие сведения
Научная работа посвящена изучению наноэлектромеханических систем при помощи методов машинного обучения. Подробно о работе можно прочитать в файле "thesis.pdf" в корневой папке.

# Работа разбита на несколько промежуточных этапов:
## Исследование сетки FEM (Mesh Research)
Идея: при слишком разреженной сетке ускоряется процесс сборки датасета, но значительно падает точность вычисляемых параметров, появляются аномалии в данных (резкие всплески, которые не могут быть обоснованы теорией); а при слишком плотной сетке процесс сборки датасета становится очень медленным, при том, что точность вычислений уже не увеличивается.

## Форматирование данных (Data Formatting)
Работа с представлением данных:
- разработаны функции, которые осуществляют поиск аномалий в датасете и их устранение.
- разработаны функции, которые приводят датасет к удобному виду

## Изучение собранных данных (Data Review)
Здесь проводится визуализация распределения всех входных и выходных параметров в датасете. Здесь проводится устранение аномальных значений параметров.

## Применение методов классического машинного обучения (Classic ML)
В данном разделе содержатся результаты применения следующих моделей:
- `LinearRegressor`
- `RandomForestRegressor`
- `TabNetRegressor` + `Optuna`
- `XGBRegressor` + `Optuna`
- `XGBRegressor` + custom `Cross Validation`

а также дополнительно визуализируется работа самодельного скейлера данных для контроля качества его работы. Проведено графическое сравнение получаемых с помощью каждой из моделей метрик.

## Применение методов нейросетевого машинного обучения (Neural ML)
В данном разделе были применены следующие модели:
#### `Fully Connected Network`, в которую входят несколько слоев `nn.Linear` и `nn.BatchNorm1d`.

#### `Branched Fully Connected Network`.
<img src="https://user-images.githubusercontent.com/112618861/229566010-5db2e9ec-8832-443f-8a31-ee2826920819.png" width="400" height="400">
(идея архитектуры нейросети и картинка взяты из работы Michelucci, Umberto, and Francesca Venturini. "Multi-task learning for multi-dimensional regression: Application to luminescence sensing." Applied Sciences 9.22 (2019): 4748., https://doi.org/10.3390/app9224748)

Сеть также состоит из слоев `nn.Linear` и `nn.BatchNorm1d`. 
Сеть делится на три части:
- Общая часть (network_general): в ней осуществляется первичная обработка батча. После нее пайплайн сети расходится на два ветки - короткую и длинную.
- Короткая ветвь (network_short): это один слой `nn.Linear`, который сжимает вектор размера `neck_features` до тензора размера `(batch_size, num_pars_y)`.
- Длинная ветвь  (network_branches): для каждого из параметров создается несколько отдельных слоев `nn.Linear` и `nn.BatchNorm1d`. В конце каждой такой длинной ветви находится слой `nn.Linear(..., out_features=1, ...)`, на выходе которого получается тензор размера `(batch_size, num_pars_y)` (по сути, предсказание одного параметра).

Особенностью сети является алгоритм вычисления ошибки вычислений и алгоритм вычисления конечного результата. Общая ошибка вычислений складывается из суммы ошибок каждой из ветвей (короткой + всех длинных) по формуле:
$$ Loss_{total} = \alpha_{short branch} \; Loss_{short branch} + \alpha_{long branch} \; Loss_{long branch} $$
а конечный результат может быть вычислен как усреднение выходов короткой и длинных ветвей с весами:
$$ output = \beta_{short branch} output_{short branch} + \beta_{short branch} output_{short branch}
(в экспериментах чаще всего брались $\beta_{short branch} = 0$ и $\beta_{long branch} = 1$). Все рассматриваемые веса являются гиперпараметрами.

#### `Branched Separate Fully Connected Network`. 
Сеть построена по такому же принципу, как и предыдущая, за исключением метода подсчета ошибки вычислении - тут складывается суммы ошибок каждой из ветвей (короткой + всех длинных) по формуле:
$$ Loss_{total} = \alpha_{short branch) \; Loss_{short branch} + \sum_{n = 0}^{N} \alpha_{n} \; Loss_{n} $$
где $N$ - число длинных ветвей (по сути, число предсказываемых параметров). Таким образом, для каждой из длинных ветвей можно менять свои веса.

#### `Wrapped Branched Fully Connected Network`
В данная сеть является модернизированной версии сети `Branched Fully Connected`. В модернизированной версии идет обучение не только самой модели, но и двух коэффициентов перед потерями короткой и длинной ветвью $\alpha_{long_branch}$ и $\alpha_{short_branch}$. Обучение этих параметров идет таким образом, чтобы 
