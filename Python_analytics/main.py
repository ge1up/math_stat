# https://www.kaggle.com/datasets/kapturovalexander/maang-share-prices-till-february-2024
# MAANG share prices till February 2024
import math

from scipy.stats import shapiro, mannwhitneyu, norm, wilcoxon
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    # 1
    data = pd.read_csv('Shares-Data.csv')
    observation_number = data.shape[0]
    print("Всего наблюдений: " + str(observation_number), '\n')
    variable_number = data.shape[1]
    print("Всего переменных: " + str(variable_number), '\n')
    print("Имя переменной - Тип данных: \n" + str(data.dtypes))
    print("\nКоличество пропущенных значений в каждом столбце:")
    missing_data = data.isnull()
    print(missing_data.sum())
    miss_count = 0
    print()
    for column in missing_data.columns:
        miss_count += missing_data[column].sum()
    print('Общее количество пропущенных значений = ' + str(miss_count))
    print('_________________________________________________________________________\n')

    # 2

    print("Основные статистические характеристики: \n")

    avr_values = []
    dispersions = []
    min_values = []
    max_values = []
    quartiles = []

    for column in data.columns:

        if column == 'Date':
            continue

        # среднее значение
        average = data[column].sum() / data[column].count()
        avr_values.append(data[column].sum() / data[column].count())

        # минимум и максимум
        min_values.append(data[column].min())
        max_values.append(data[column].max())

        # дисперсия
        dispersion = 0

        for unit in data[column]:
            dispersion += pow(unit - average, 2)

        dispersions.append(pow((dispersion / observation_number), 2))

        # квартили
        q1 = data[column].quantile(0.25)
        q2 = data[column].quantile(0.5)
        q3 = data[column].quantile(0.75)
        quartiles.append([q1, q2, q3])

    characteristics = {
        "avr_values": avr_values,
        "dispersions": dispersions,
        "min_values": min_values,
        "max_values": max_values,
        "quartiles": quartiles
    }

    columns_name = [x for x in data.columns if x != 'Date']

    print(pd.DataFrame(characteristics, index=columns_name).to_string(), '\n')

    # корреляции

    data_numeric = data.drop(columns=['Date'])

    # Вычисление матрицы корреляций
    correlation_matrix = data_numeric.corr()

    # Вывод матрицы корреляций
    print("Матрица корреляций:")
    print(correlation_matrix)

    print('\n___________________________________________________________________________\n')

    # 3

    print(
        'Гистограммы с графиком функции распределения, боксплоты и диаграммы рассеивания представлены в виде графиков\n')

    for column in columns_name:

        fig = plt.figure()

        # Создаем три графика
        ax1 = fig.add_axes([0.05, 0.1, 0.3, 0.8])  # Размеры и положение для первого графика
        ax2 = fig.add_axes([0.4, 0.1, 0.25, 0.8])  # Размеры и положение для второго графика
        ax3 = fig.add_axes([0.7, 0.1, 0.25, 0.8])  # Размеры и положение для третьего графика

        # Построение гистограммы функции распределения

        values = data[column].values
        sorted_values = np.sort(values)
        cumulative_distribution = np.arange(len(sorted_values)) / len(values)

        # sns.histplot(sorted_values, bins=30, stat='density', cumulative=True, color='skyblue', edgecolor='black',
        #              alpha=0.7, ax=ax1)
        sns.histplot(data=pd.DataFrame(data[column]), kde=True, ax=ax1)
        ax1.set_title('Гистограмма функции распределения для столбца ' + column)
        ax1.set_xlabel('Значение параметра ' + column)
        ax1.set_ylabel('Кол-во элементов с таким значением')
        ax1.grid(True)

        # Построение боксплота

        ax2.boxplot(data[column], vert=False)
        ax2.set_title('Боксплот для столбца ' + column)
        ax2.set_xlabel('Значение ' + column)
        ax2.grid(True)

        # Построение диаграммы рассеивания с параметром Volume

        if column != 'Volume':
            ax3.scatter(data[column], data['Volume'], color='blue', alpha=0.5)
            ax3.set_title('Диаграмма рассеивания для столбца ' + column)
            ax3.set_xlabel(column)
            ax3.set_ylabel('Volume')
            ax3.grid(True)

        fig.set_size_inches(20, 10)
        fig.savefig(column + '.png', dpi=100)
        plt.close()

    print('__________________________________________________________________________\n')

    # 4

    print('Выбросы:\n')

    bad_numbers = set()

    for column in columns_name:
        # Вычисление квартилей
        Q1 = np.percentile(data[column], 25)
        Q3 = np.percentile(data[column], 75)

        IQR = Q3 - Q1

        # Вычисление границ для определения выбросов
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Вывод кол-ва выбросов по столбцам и сохранение индексов выбросов
        no_emissions = [index for index, value in enumerate(data[column]) if value < lower_bound or value > upper_bound]
        print('В столбце ' + column + ' ' + str(len(no_emissions)) + ' выбросов.')
        bad_numbers.update(no_emissions)

    correct_data = {column: [] for column in data.columns}
    for column in data.columns:
        for index, value in enumerate(data[column]):
            if index not in bad_numbers:
                correct_data[column].append(value)

    correct_data = pd.DataFrame(correct_data)

    print('\nВсего наблюдений, где хотя бы один параметр являлся бы выброом - ' + str(
        correct_data.shape[0] - data.shape[0]))
    print('Размер dataframe без выбросов - ' + str(correct_data.shape[0]))
    print('\n_________________________________________________________________________\n')

    # 5

    # Подготовка

    print('Предположим, что наши данные распределены нормально.')
    print('Выдвинем гипотезу H0 о том, что распределение данных нормальное.')
    print('H1 - альтернативная гипотеза - будет говорить об обратном.')
    print('С помощью теста Шапиро-Уилка проверим гипотезу:\n')

    statistic, p_value = shapiro(data['Volume'])

    # Вывод результатов
    print("Статистика теста:", statistic)
    print("p-значение:", p_value)

    print()

    alpha = 0.05
    if p_value > alpha:
        print("Не удалось отвергнуть нулевую гипотезу: данные могут быть нормально распределены")
    else:
        print("Отвергаем нулевую гипотезу: данные не являются нормально распределенными\n")

    print('__________________________________________________________________________\n')

    print('Выдвинем гипотезу о том, что среднее кол-во операций на бирже в год до ковида и после не отличается')
    print('Уровень значимости(альфа) возьмем за 0.05\n')

    # Фильтрация данных для 2018 и 2020 годов
    data_2018 = data[data['Date'].str.contains('2018')]
    data_2020 = data[data['Date'].str.contains('2020')]

    # Извлечение значений Volume для 2018 и 2020 годов
    volume_2018 = data_2018['Volume'].tolist()
    volume_2020 = data_2020['Volume'].tolist()[:251]

    # Проведение t-теста
    statistic, p_value = wilcoxon(volume_2018, volume_2020)

    # Вывод результатов
    print("Статистика теста(наблюдаемое значение):", statistic)
    print("p-значение:", p_value)

    # Оценка статистической значимости
    alpha = 0.05
    if p_value < alpha:
        print("Различия в объеме сделок статистически значимы (отвергаем нулевую гипотезу)")
    else:
        print("Нет статистически значимых различий в объеме сделок (нет оснований отвергать нулевую гипотезу)")

    print('\n__________________________________________________________________________\n')

    # 6

    print('1 тест\n')

    print('Проведем построение линейной регресси по переменной Open как независимой переменной и Close как зависимой\n')

    # Выбираем зависимую переменную (Close) и независимую переменную (Open)
    open = data[['Open']]  # Независимая переменная
    close = data['Close']  # Зависимая переменная

    # Создаем и обучаем модель линейной регрессии
    model = LinearRegression()
    model.fit(open, close)

    # Выводим коэффициенты модели
    print('Intercept:', model.intercept_)  # пересечение
    print('Coefficient:', model.coef_)  # коэффициент наклона

    print('\nЭто означает, что у нас есть следующая линейная зависимость: Close =', model.coef_, '* Open +',
          model.intercept_, '\n')

    print('Таким образом, каждый дополнительный единичный прирост в переменной Open приведет'
          ' к увеличению переменной Close на примерно 1.00062685 единиц, при условии, что все остальные факторы остаются константными')

    print('\nКоээфициент корреляции -', data['Open'].corr(data['Close']))
    print('С точки зрения связи мы получаем сильную положительную связь между группами')

    # Предсказываем значения Close на основе данных по Open
    y_pred = model.predict(open)

    # Визуализируем результаты
    plt.scatter(open, close, color='blue', label='Actual')
    plt.plot(open, y_pred, color='red', linewidth=2, label='Predicted')
    plt.xlabel('Open')
    plt.ylabel('Close')
    plt.title('Linear Regression: Open vs Close')
    plt.legend()
    plt.show()

    print('\n2 тест\n')

    print('Проведем построение линейной регресси по разнице между наибольшей '
          'и наименьшой ценой продажи акции как независимой переменной и переменной Volume как зависимой\n')

    difference = pd.DataFrame(data['High'] - data['Low'])  # Независимая переменная
    volume = data['Volume']  # Зависимая переменная

    model = LinearRegression()
    model.fit(difference, volume)

    # Выводим коэффициенты модели
    print('Intercept:', model.intercept_)  # пересечение
    print('Coefficient:', model.coef_)  # коэффициент наклона

    print('\nЭто означает, что у нас есть следующая линейная зависимость: Volume =', model.coef_, '* Difference +',
          model.intercept_, '\n')

    print('Таким образом, каждый дополнительный единичный прирост в переменной Difference приведет'
          ' к увеличению переменной Volume на примерно -40915630.44188645 единиц, при условии, что все остальные факторы остаются константными')

    data['diff'] = data['High'] - data['Low']
    print('\nКоээфициент корреляции -', data['diff'].corr(data['Close']))
    print('С точки зрения связи мы получаем сильную положительную связь между группами')

    # Предсказываем значения Volume на основе данных по difference
    y_pred = model.predict(difference)

    # Визуализируем результаты
    plt.scatter(difference, volume, color='blue', label='Actual')
    plt.plot(difference, y_pred, color='red', linewidth=2, label='Predicted')
    plt.xlabel('Difference between High and Low')
    plt.ylabel('Volume')
    plt.title('Linear Regression: Difference between High and Low vs Volume')
    plt.legend()
    plt.show()