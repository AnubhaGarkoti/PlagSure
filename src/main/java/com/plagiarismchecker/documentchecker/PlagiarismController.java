package com.plagiarismchecker.documentchecker;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;
import java.util.HashMap;

@RestController
public class PlagiarismController {

    private final SemanticSimilarityService semanticService;

    public PlagiarismController(
            SemanticSimilarityService semanticService) {
        this.semanticService = semanticService;
    }

    @PostMapping("/api/compare")
    public Map<String, Object> compareDocuments(
            @RequestBody Map<String, String> documents) {

        String text1 = documents.getOrDefault("text1", "");
        String text2 = documents.getOrDefault("text2", "");

        // LCS-based textual similarity
        int lcsLength = findLCS(text1, text2);

        int words1 = countWords(text1);
        int words2 = countWords(text2);

        int maxWords = Math.max(words1, words2);

        double textualSimilarity = maxWords == 0
                ? 0.0
                : (lcsLength * 100.0) / maxWords;

        // AI semantic similarity
        double semanticSimilarity =
                semanticService.calculateSimilarity(text1, text2) * 100.0;

        // Combined similarity score
        double combinedScore =
                (textualSimilarity * 0.4)
                        + (semanticSimilarity * 0.6);

        String interpretation;

        if (combinedScore < 30) {
            interpretation = "Low similarity";
        } else if (combinedScore < 70) {
            interpretation = "Moderate similarity";
        } else {
            interpretation = "High similarity";
        }

        Map<String, Object> result = new HashMap<>();

        result.put(
                "textualSimilarity",
                Math.round(textualSimilarity * 100.0) / 100.0
        );

        result.put(
                "semanticSimilarity",
                Math.round(semanticSimilarity * 100.0) / 100.0
        );

        result.put(
                "combinedScore",
                Math.round(combinedScore * 100.0) / 100.0
        );

        result.put("lcsLength", lcsLength);
        result.put("interpretation", interpretation);

        return result;
    }

    private int countWords(String text) {

        if (text == null || text.trim().isEmpty()) {
            return 0;
        }

        return text.trim().split("\\s+").length;
    }

    private int findLCS(String text1, String text2) {

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

                    dp[i][j] =
                            dp[i - 1][j - 1] + 1;

                } else {

                    dp[i][j] =
                            Math.max(
                                    dp[i - 1][j],
                                    dp[i][j - 1]
                            );
                }
            }
        }

        return dp[m][n];
    }
}