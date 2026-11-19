package com.plagiarismchecker.documentchecker;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.apache.poi.xwpf.extractor.XWPFWordExtractor;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

@RestController
@RequestMapping("/api/documents")
public class DocumentUploadController {

    @PostMapping("/extract")
    public ResponseEntity<String> extractText(
            @RequestParam("file") MultipartFile file) {

        if (file.isEmpty()) {
            return ResponseEntity.badRequest()
                    .body("Please select a document.");
        }

        try {
            String text = extractTextFromFile(file);

            // Detect image/scanned PDFs or files with no extractable text
            if (text == null || text.trim().isEmpty()) {
                return ResponseEntity.badRequest()
                        .body("This document does not contain extractable text. "
                                + "The PDF may be scanned or image-based.");
            }

            return ResponseEntity.ok(text);

        } catch (Exception e) {
            return ResponseEntity.badRequest()
                    .body("Could not extract text from the document: "
                            + e.getMessage());
        }
    }

    private String extractTextFromFile(MultipartFile file)
            throws IOException {

        String fileName = file.getOriginalFilename();

        if (fileName == null) {
            throw new IOException("File name is missing.");
        }

        String lowerCaseName = fileName.toLowerCase();

        if (lowerCaseName.endsWith(".txt")) {

            return new String(
                    file.getBytes(),
                    StandardCharsets.UTF_8
            );

        } else if (lowerCaseName.endsWith(".pdf")) {

            try (PDDocument document =
                         Loader.loadPDF(file.getBytes())) {

                PDFTextStripper stripper =
                        new PDFTextStripper();

                return stripper.getText(document);
            }

        } else if (lowerCaseName.endsWith(".docx")) {

            try (XWPFDocument document =
                         new XWPFDocument(file.getInputStream());

                 XWPFWordExtractor extractor =
                         new XWPFWordExtractor(document)) {

                return extractor.getText();
            }

        } else {
            throw new IOException(
                    "Unsupported file format. Please upload TXT, PDF, or DOCX."
            );
        }
    }
}