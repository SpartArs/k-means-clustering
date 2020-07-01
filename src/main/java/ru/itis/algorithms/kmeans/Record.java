package ru.itis.algorithms.kmeans;

import java.util.Map;
import java.util.Objects;

/**
 * Инкапсулирует все значения объектов для нескольких атрибутов.
 * При необходимости каждая запись может быть описана с помощью поля {@link #description}.
 */
public class Record {

    /**
     * Описание записи.
     */
    private final String description;

    /**
     * Инкапсулирует все атрибуты и соответствующие им значения, то есть свойства.
     */
    private final Map<String, Double> features;

    public Record(String description, Map<String, Double> features) {
        this.description = description;
        this.features = features;
    }

    public Record(Map<String, Double> features) {
        this("", features);
    }

    public String getDescription() {
        return description;
    }

    public Map<String, Double> getFeatures() {
        return features;
    }

    @Override
    public String toString() {
        String prefix = description == null || description
                .trim()
                .isEmpty() ? "Record" : description;

        return prefix + ": " + features;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Record record = (Record) o;
        return Objects.equals(getDescription(), record.getDescription()) && Objects.equals(getFeatures(), record.getFeatures());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getDescription(), getFeatures());
    }
}
