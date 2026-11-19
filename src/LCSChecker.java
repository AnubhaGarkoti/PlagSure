import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Scanner;

public class LCSChecker {

    // =========================
    // LCS
    // =========================

    public static int findLCS(String text1, String text2) {

        String[] words1 = text1.toLowerCase().trim().isEmpty()
                ? new String[0]
                : text1.toLowerCase().trim().split("\\s+");

        String[] words2 = text2.toLowerCase().trim().isEmpty()
                ? new String[0]
                : text2.toLowerCase().trim().split("\\s+");

        int m = words1.length;
        int n = words2.length;

        int[][] dp = new int[m + 1][n + 1];

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {

                if (words1[i - 1].equals(words2[j - 1])) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(
                            dp[i - 1][j],
                            dp[i][j - 1]
                    );
                }
            }
        }

        return dp[m][n];
    }

    // =========================
    // LCS TEXTUAL SIMILARITY
    // =========================

    public static double calculateSimilarity(
            String text1,
            String text2) {

        if (text1.trim().isEmpty()
                || text2.trim().isEmpty()) {
            return 0.0;
        }

        int lcsLength = findLCS(text1, text2);

        int words1 =
                text1.trim().split("\\s+").length;

        int words2 =
                text2.trim().split("\\s+").length;

        int maxLength =
                Math.max(words1, words2);

        return (lcsLength * 100.0) / maxLength;
    }

    // =========================
    // GEMINI SEMANTIC SIMILARITY
    // =========================

    public static double calculateSemanticSimilarity(
            String text1,
            String text2,
            String apiKey) {

        try {

            String url =
                    "https://generativelanguage.googleapis.com/"
                    + "v1beta/models/gemini-embedding-001:embedContent"
                    + "?key=" + apiKey;

            double[] embedding1 =
                    getEmbedding(text1, apiKey);

            double[] embedding2 =
                    getEmbedding(text2, apiKey);

            if (embedding1 == null || embedding2 == null) {
                return 0.0;
            }

            return cosineSimilarity(
                    embedding1,
                    embedding2
            ) * 100.0;

        } catch (Exception e) {

            System.out.println(
                    "\nGemini semantic similarity error: "
                    + e.getMessage()
            );

            return 0.0;
        }
    }

    // =========================
    // GET GEMINI EMBEDDING
    // =========================

    private static double[] getEmbedding(
            String text,
            String apiKey) throws Exception {

        String url =
                "https://generativelanguage.googleapis.com/"
                + "v1beta/models/gemini-embedding-001:embedContent"
                + "?key=" + apiKey;

        String jsonBody =
                "{"
                + "\"model\":\"models/gemini-embedding-001\","
                + "\"content\":{"
                + "\"parts\":["
                + "{"
                + "\"text\":\""
                + escapeJson(text)
                + "\""
                + "}"
                + "]"
                + "},"
                + "\"taskType\":\"SEMANTIC_SIMILARITY\""
                + "}";

        HttpClient client =
                HttpClient.newHttpClient();

        HttpRequest request =
                HttpRequest.newBuilder()
                        .uri(URI.create(url))
                        .header(
                                "Content-Type",
                                "application/json"
                        )
                        .POST(
                                HttpRequest.BodyPublishers
                                        .ofString(jsonBody)
                        )
                        .build();

        HttpResponse<String> response =
                client.send(
                        request,
                        HttpResponse.BodyHandlers.ofString()
                );

        if (response.statusCode() != 200) {

            System.out.println(
                    "\nGemini API error: "
                    + response.statusCode()
            );

            System.out.println(
                    "Response: "
                    + response.body()
            );

            return null;
        }

        return parseEmbedding(
                response.body()
        );
    }

    // =========================
    // PARSE EMBEDDING
    // =========================

    private static double[] parseEmbedding(
            String json) {

        int valuesStart =
                json.indexOf("\"values\"");

        if (valuesStart == -1) {
            return null;
        }

        int start =
                json.indexOf(
                        "[",
                        valuesStart
                );

        int end =
                json.indexOf(
                        "]",
                        start
                );

        if (start == -1 || end == -1) {
            return null;
        }

        String numbers =
                json.substring(
                        start + 1,
                        end
                );

        String[] parts =
                numbers.split(",");

        double[] embedding =
                new double[parts.length];

        for (int i = 0;
             i < parts.length;
             i++) {

            embedding[i] =
                    Double.parseDouble(
                            parts[i].trim()
                    );
        }

        return embedding;
    }

    // =========================
    // COSINE SIMILARITY
    // =========================

    private static double cosineSimilarity(
            double[] vector1,
            double[] vector2) {

        if (vector1.length != vector2.length) {
            return 0.0;
        }

        double dotProduct = 0.0;
        double magnitude1 = 0.0;
        double magnitude2 = 0.0;

        for (int i = 0;
             i < vector1.length;
             i++) {

            dotProduct +=
                    vector1[i] * vector2[i];

            magnitude1 +=
                    vector1[i] * vector1[i];

            magnitude2 +=
                    vector2[i] * vector2[i];
        }

        double denominator =
                Math.sqrt(magnitude1)
                * Math.sqrt(magnitude2);

        if (denominator == 0) {
            return 0.0;
        }

        return dotProduct / denominator;
    }

    // =========================
    // JSON ESCAPING
    // =========================

    private static String escapeJson(
            String text) {

        return text
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    // =========================
    // COMBINED SCORE
    // =========================

    public static double calculateCombinedScore(
            double lcsSimilarity,
            double semanticSimilarity) {

        // 40% LCS + 60% AI Semantic Similarity
        return (lcsSimilarity * 0.40)
                + (semanticSimilarity * 0.60);
    }

    // =========================
    // MAIN
    // =========================

    public static void main(String[] args) {

        Scanner scanner =
                new Scanner(System.in);

        String apiKey =
                System.getenv("GEMINI_API_KEY");

        if (apiKey == null
                || apiKey.trim().isEmpty()) {

            System.out.println(
                    "ERROR: Gemini API key not found."
            );

            System.out.println(
                    "Please set GEMINI_API_KEY "
                    + "in the terminal."
            );

            scanner.close();
            return;
        }

        System.out.println(
                "=========================================="
        );

        System.out.println(
                "       DOCUMENT PLAGIARISM CHECKER"
        );

        System.out.println(
                "=========================================="
        );

        System.out.println(
                "\nEnter text for Document 1:"
        );

        String document1 =
                scanner.nextLine();

        System.out.println(
                "\nEnter text for Document 2:"
        );

        String document2 =
                scanner.nextLine();

        if (document1.trim().isEmpty()
                || document2.trim().isEmpty()) {

            System.out.println(
                    "\nBoth documents must contain text."
            );

            scanner.close();
            return;
        }

        // LCS
        int lcs =
                findLCS(
                        document1,
                        document2
                );

        double lcsSimilarity =
                calculateSimilarity(
                        document1,
                        document2
                );

        // AI semantic similarity
        System.out.println(
                "\nCalculating AI semantic similarity..."
        );

        double semanticSimilarity =
                calculateSemanticSimilarity(
                        document1,
                        document2,
                        apiKey
                );

        // Combined score
        double combinedScore =
                calculateCombinedScore(
                        lcsSimilarity,
                        semanticSimilarity
                );

        // Results
        System.out.println(
                "\n=========================================="
        );

        System.out.println(
                "                RESULTS"
        );

        System.out.println(
                "=========================================="
        );

        System.out.println(
                "LCS Length: "
                + lcs
        );

        System.out.printf(
                "LCS Textual Similarity: %.2f%%%n",
                lcsSimilarity
        );

        System.out.printf(
                "AI Semantic Similarity: %.2f%%%n",
                semanticSimilarity
        );

        System.out.printf(
                "Combined Similarity Score: %.2f%%%n",
                combinedScore
        );

        // Classification
        if (combinedScore < 30) {

            System.out.println(
                    "Result: Low similarity"
            );

        } else if (combinedScore < 70) {

            System.out.println(
                    "Result: Moderate similarity"
            );

        } else {

            System.out.println(
                    "Result: High similarity"
            );
        }

        System.out.println(
                "\nNote: This score is an indicator "
                + "of similarity and is not proof of plagiarism."
        );

        System.out.println(
                "=========================================="
        );

        scanner.close();
    }
}