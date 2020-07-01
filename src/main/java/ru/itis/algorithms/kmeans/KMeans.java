package ru.itis.algorithms.kmeans;

import java.util.*;

import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toSet;

/**
 * Инкапсулирует реализацию алгоритма кластеризации k-means.
 */
public class KMeans {

    /**
     * Будет использоваться для генерации случайных чисел.
     */
    private static final Random random = new Random();

    /**
     * @param records       Набор данных.
     * @param k             Количество кластеров.
     * @param distance      Для расчёта расстояния между двумя элементами.
     * @param maxIterations Максимальное число итераций.
     * @return K кластеры вместе с их свойствами.
     */
    public static Map<Centroid, List<Record>> fit(List<Record> records, int k, Distance distance, int maxIterations) {
        applyPreconditions(records, k, distance, maxIterations);
        List<Centroid> centroids = randomCentroids(records, k);
        Map<Centroid, List<Record>> clusters = new HashMap<>();
        Map<Centroid, List<Record>> lastState = new HashMap<>();

        // итерация для заранее определенного количества раз
        for (int i = 0; i < maxIterations; i++) {
            boolean isLastIteration = i == maxIterations - 1;

            // в каждой итерации мы должны найти ближайший центроид для каждой записи
            for (Record record : records) {
                Centroid centroid = nearestCentroid(record, centroids, distance);
                assignToCluster(clusters, record, centroid);
            }

            // если назначение не изменяется, то алгоритм завершается
            boolean shouldTerminate = isLastIteration || clusters.equals(lastState);
            lastState = clusters;
            if (shouldTerminate) {
                break;
            }

            // в конце каждой итерации мы должны переместить центроиды
            centroids = relocateCentroids(clusters);
            clusters = new HashMap<>();
        }

        return lastState;
    }

    /**
     * Перемещает все центроиды в среднее значение всех назначенных свойств.
     *
     * @param clusters Текущая конфигурация кластера.
     * @return Коллекция новых и перемещённых центроидов.
     */
    private static List<Centroid> relocateCentroids(Map<Centroid, List<Record>> clusters) {
        return clusters
                .entrySet()
                .stream()
                .map(e -> average(e.getKey(), e.getValue()))
                .collect(toList());
    }

    /**
     * Перемещает данный центроид в среднее положение всех назначенных объектов.
     * Если центроид не имеет никакой функции в своем кластере,то не было бы никакой необходимости в перемещении.
     * В противном случае для каждой записи мы вычисляем среднее значение всех записей, сначала суммируя все записи,
     * а затем деля итоговое значение суммирования на количество записей.
     *
     * @param centroid Центроид для перемещения.
     * @param records  Назначенные свойства.
     * @return Перемещённый центроид.
     */
    private static Centroid average(Centroid centroid, List<Record> records) {
        // если этот кластер пуст, то мы не должны перемещать центроид
        if (records == null || records.isEmpty()) {
            return centroid;
        }

        // Поскольку некоторые записи не имеют всех возможных атрибутов,
        // мы инициализируем средние координаты, равные текущим координатам центроида
        Map<String, Double> average = centroid.getCoordinates();

        // Функция average работает корректно, если мы очистим все координаты, соответствующие текущим атрибутам записи
        records
                .stream()
                .flatMap(e -> e
                        .getFeatures()
                        .keySet()
                        .stream())
                .forEach(k -> average.put(k, 0.0));

        for (Record record : records) {
            record
                    .getFeatures()
                    .forEach((k, v) -> average.compute(k, (k1, currentValue) -> v + currentValue));
        }

        average.forEach((k, v) -> average.put(k, v / records.size()));

        return new Centroid(average);
    }

    /**
     * Присваивает вектор свойств данному центроиду.
     * Если это первое задание для данного центроида, то сначала мы должны создать список.
     *
     * @param clusters Текущая конфигурация кластера.
     * @param record   Вектор свойтсв.
     * @param centroid Центроид.
     */
    private static void assignToCluster(Map<Centroid, List<Record>> clusters, Record record, Centroid centroid) {
        clusters.compute(centroid, (key, list) -> {
            if (list == null) {
                list = new ArrayList<>();
            }

            list.add(record);
            return list;
        });
    }

    /**
     * С помощью данного расчёта расстояний перебирает центроиды и находит ближайший к данной записи.
     *
     * @param record    Вектор свойств, для которого нужно найти центроид.
     * @param centroids Коллекция всех центроидов.
     * @param distance  Для расчёта расстояния между двумя предметами.
     * @return Ближайший центроид к данному вектору свойств.
     */
    private static Centroid nearestCentroid(Record record, List<Centroid> centroids, Distance distance) {
        double minimumDistance = Double.MAX_VALUE;
        Centroid nearest = null;

        for (Centroid centroid : centroids) {
            double currentDistance = distance.calculate(record.getFeatures(), centroid.getCoordinates());

            if (currentDistance < minimumDistance) {
                minimumDistance = currentDistance;
                nearest = centroid;
            }
        }

        return nearest;
    }

    /**
     * Генерирует k случайных центроидов.
     * Прежде чем начать процесс генерации центроидов, сначала мы рассчитаем возможный диапазон значений для каждого атрибута.
     * Затем, когда мы собираемся генерировать центроиды, мы генерируем случайные координаты в диапазоне [min, max] для каждого атрибута.
     *
     * @param records Набор данных, который помогает вычислить диапазон [min, max] для каждого атрибута.
     * @param k       Количество кластеров.
     * @return Коллекции случайно сгенерированных центроидов.
     */
    private static List<Centroid> randomCentroids(List<Record> records, int k) {
        List<Centroid> centroids = new ArrayList<>();
        Map<String, Double> maxs = new HashMap<>();
        Map<String, Double> mins = new HashMap<>();
        for (Record record : records) {
            record
                    .getFeatures()
                    .forEach((key, value) -> {
                        // сравнивает значение с текущим максимумом и выбирает между ними большее значение
                        maxs.compute(key, (k1, max) -> max == null || value > max ? value : max);

                        // сравнивает значение с текущим min и выбирает меньшее значение между ними
                        mins.compute(key, (k1, min) -> min == null || value < min ? value : min);
                    });
        }
        Set<String> attributes = records
                .stream()
                .flatMap(e -> e
                        .getFeatures()
                        .keySet()
                        .stream())
                .collect(toSet());
        for (int i = 0; i < k; i++) {
            Map<String, Double> coordinates = new HashMap<>();
            for (String attribute : attributes) {
                double max = maxs.get(attribute);
                double min = mins.get(attribute);
                coordinates.put(attribute, random.nextDouble() * (max - min) + min);
            }

            centroids.add(new Centroid(coordinates));
        }

        return centroids;
    }

    private static void applyPreconditions(List<Record> records, int k, Distance distance, int maxIterations) {
        if (records == null || records.isEmpty()) {
            throw new IllegalArgumentException("Датасет не должен быть пустым!");
        }

        if (k <= 1) {
            throw new IllegalArgumentException("Количество кластеров должно быть больше 1!");
        }

        if (distance == null) {
            throw new IllegalArgumentException("Необходим класс расчёта расстояния!");
        }

        if (maxIterations <= 0) {
            throw new IllegalArgumentException("Максимальное число итераций должно быть положительным числом!");
        }
    }
}
