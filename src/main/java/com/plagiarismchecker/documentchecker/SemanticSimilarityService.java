package com.plagiarismchecker.documentchecker;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.*;

import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;

@Service
public class SemanticSimilarityService {

    private final OrtEnvironment environment;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;

    public SemanticSimilarityService() throws Exception {

        environment = OrtEnvironment.getEnvironment();

        // Load ONNX model directly from the JAR/classpath
        InputStream modelStream =
                getClass().getClassLoader()
                        .getResourceAsStream("models/model.onnx");

        if (modelStream == null) {
            throw new RuntimeException(
                    "models/model.onnx not found in classpath."
            );
        }

        byte[] modelBytes;

        try (modelStream) {
            modelBytes = modelStream.readAllBytes();
        }

        session = environment.createSession(
                modelBytes,
                new OrtSession.SessionOptions()
        );

        // Load tokenizer directly from the JAR/classpath
        InputStream tokenizerStream =
                getClass().getClassLoader()
                        .getResourceAsStream("models/tokenizer.json");

        if (tokenizerStream == null) {
            throw new RuntimeException(
                    "models/tokenizer.json not found in classpath."
            );
        }

        try (tokenizerStream) {
            tokenizer =
                    HuggingFaceTokenizer.newInstance(
                            tokenizerStream,
                            null
                    );
        }
    }

    public double calculateSimilarity(
            String text1,
            String text2) {

        try {

            float[] embedding1 =
                    generateEmbedding(text1);

            float[] embedding2 =
                    generateEmbedding(text2);

            return cosineSimilarity(
                    embedding1,
                    embedding2
            );

        } catch (Exception e) {

            System.err.println(
                    "SEMANTIC SIMILARITY ERROR:"
            );

            e.printStackTrace();

            return 0.0;
        }
    }

    private float[] generateEmbedding(
            String text) throws Exception {

        long[] tokenIds =
                tokenizer.encode(text)
                        .getIds();

        int maxLength = 128;

        long[] inputIds =
                new long[maxLength];

        long[] attentionMask =
                new long[maxLength];

        int length =
                Math.min(
                        tokenIds.length,
                        maxLength
                );

        for (int i = 0; i < length; i++) {

            inputIds[i] =
                    tokenIds[i];

            attentionMask[i] = 1;
        }

        long[] shape = {
                1,
                maxLength
        };

        OnnxTensor inputIdsTensor =
                OnnxTensor.createTensor(
                        environment,
                        LongBuffer.wrap(inputIds),
                        shape
                );

        OnnxTensor attentionTensor =
                OnnxTensor.createTensor(
                        environment,
                        LongBuffer.wrap(attentionMask),
                        shape
                );

        OnnxTensor tokenTypeTensor =
                OnnxTensor.createTensor(
                        environment,
                        LongBuffer.wrap(
                                new long[maxLength]
                        ),
                        shape
                );

        try (
                inputIdsTensor;
                attentionTensor;
                tokenTypeTensor
        ) {

            Map<String, OnnxTensor> inputs =
                    new HashMap<>();

            inputs.put(
                    "input_ids",
                    inputIdsTensor
            );

            inputs.put(
                    "attention_mask",
                    attentionTensor
            );

            inputs.put(
                    "token_type_ids",
                    tokenTypeTensor
            );

            try (OrtSession.Result result =
                         session.run(inputs)) {

                Object value =
                        result.get(0).getValue();

                float[][][] output =
                        (float[][][]) value;

                float[][] tokenEmbeddings =
                        output[0];

                int dimensions =
                        tokenEmbeddings[0].length;

                float[] sentenceEmbedding =
                        new float[dimensions];

                int validTokens = 0;

                for (int i = 0;
                     i < length;
                     i++) {

                    if (attentionMask[i] == 1) {

                        for (int j = 0;
                             j < dimensions;
                             j++) {

                            sentenceEmbedding[j] +=
                                    tokenEmbeddings[i][j];
                        }

                        validTokens++;
                    }
                }

                for (int j = 0;
                     j < dimensions;
                     j++) {

                    sentenceEmbedding[j] /=
                            Math.max(
                                    validTokens,
                                    1
                            );
                }

                return sentenceEmbedding;
            }
        }
    }

    private double cosineSimilarity(
            float[] vector1,
            float[] vector2) {

        double dotProduct = 0.0;
        double magnitude1 = 0.0;
        double magnitude2 = 0.0;

        int length =
                Math.min(
                        vector1.length,
                        vector2.length
                );

        for (int i = 0;
             i < length;
             i++) {

            dotProduct +=
                    vector1[i] * vector2[i];

            magnitude1 +=
                    vector1[i] * vector1[i];

            magnitude2 +=
                    vector2[i] * vector2[i];
        }

        if (magnitude1 == 0 ||
                magnitude2 == 0) {

            return 0.0;
        }

        return dotProduct /
                (Math.sqrt(magnitude1) *
                        Math.sqrt(magnitude2));
    }
}