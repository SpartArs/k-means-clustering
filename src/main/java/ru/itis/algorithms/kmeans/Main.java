package ru.itis.algorithms.kmeans;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import static java.util.stream.Collectors.toSet;

public class Main {

    public static void main(String[] args) throws IOException {
        Properties property = getProperties();
        String fileName = property.getProperty("file.name");
        int clustersCount = Integer.parseInt(property.getProperty("clusters.count"));
        int maxIterations = Integer.parseInt(property.getProperty("iterations.max"));

        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        List<Record> records = new ArrayList<>();
        String firstLine = reader.readLine();
        List<String> headers = new ArrayList<>(Arrays.asList(firstLine.split(",")));
        String line = null;

        while ((line = reader.readLine()) != null) {
            Map<String, Double> features = new HashMap<>();
            List<String> nextLine = new ArrayList(Arrays.asList(line.split(",")));
            if (headers.size() == nextLine.size()) {
                for (int i = 0; i < headers.size(); i++) {
                    features.put(headers.get(i), Double.parseDouble(nextLine.get(i)));
                }
            }

            records.add(new Record(features));
        }

        reader.close();

        Map<Centroid, List<Record>> clusters = KMeans.fit(records, clustersCount, new EuclideanDistance(), maxIterations);

        int i = 0;
        clusters.forEach((key, value) -> {
            System.out.println("------------------------------ CLUSTER -----------------------------------");

            System.out.println(sortedCentroid(key));
            String members = String.join(", ", value
                    .stream()
                    .map(Record::getDescription)
                    .collect(toSet()));
            System.out.print(members);

            System.out.println();
            System.out.println();
        });
    }

    private static Centroid sortedCentroid(Centroid key) {
        List<Map.Entry<String, Double>> entries = new ArrayList<>(key
                .getCoordinates()
                .entrySet());
        entries.sort((e1, e2) -> e2
                .getValue()
                .compareTo(e1.getValue()));

        Map<String, Double> sorted = new LinkedHashMap<>();
        for (Map.Entry<String, Double> entry : entries) {
            sorted.put(entry.getKey(), entry.getValue());
        }

        return new Centroid(sorted);
    }

    private static Properties getProperties() throws IOException {
        FileInputStream fileInputStream = new FileInputStream("src/main/resources/config.properties");
        Properties property = new Properties();
        property.load(fileInputStream);

        return property;
    }

}