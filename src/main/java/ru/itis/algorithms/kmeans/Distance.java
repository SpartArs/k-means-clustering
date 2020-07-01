package ru.itis.algorithms.kmeans;

import java.util.Map;

/**
 * Определяет контракт для вычисления расстояния между двумя векторами объектов.
 * Чем меньше расчетное расстояние, тем больше два элемента похожи друг на друга.
 */
public interface Distance {

    /**
     * @param f1 Первый набор функций.
     * @param f2 Второй набор функций.
     * @return Расчетное расстояние.
     * @throws IllegalArgumentException Если заданные векторы свойств недопустимы.
     */
    double calculate(Map<String, Double> f1, Map<String, Double> f2);
}
