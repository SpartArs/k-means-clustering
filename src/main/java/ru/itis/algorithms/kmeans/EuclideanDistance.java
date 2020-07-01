package ru.itis.algorithms.kmeans;

import java.util.Map;

/**
 * Вычисляет расстояние между двумя элементами с помощью евклидовой формулы.
 */
public class EuclideanDistance implements Distance {

    @Override
    public double calculate(Map<String, Double> f1, Map<String, Double> f2) {
        if (f1 == null || f2 == null) {
            throw new IllegalArgumentException("Векторы свойств не могут быть нулевыми!");
        }

        double sum = 0;
        for (String key : f1.keySet()) {
            Double v1 = f1.get(key);
            Double v2 = f2.get(key);

            if (v1 != null && v2 != null) sum += Math.pow(v1 - v2, 2);
        }

        return Math.sqrt(sum);
    }
}
