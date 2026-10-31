package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strings"
)

// Punctuation and encoding fixes applied before splitting.
var punctuationAndEncodingReplacer = strings.NewReplacer(
	"’", "'", // curly apostrophe
	"‘", "'", // curly open quote
	"—", "-", // em dash
	"–", "-", // en dash
	"æ", "ae",
	"à", "a",
	"&c", "etc",
)

// One word (letters/digits, apostrophes kept inside) or one punctuation mark.
// Underscores are deliberately absent, so leftover stage-direction markers
// like "_Hic" become plain words.
var wordOrPunctuationPattern = regexp.MustCompile(`[a-z0-9]+(?:'[a-z0-9]+)*|[.,;:!?&"()\[\]-]`)

// NormalizeAndSplitToken lowercases one raw chunk and splits it into tokens.
func NormalizeAndSplitToken(rawToken string) []string {
	lowercasedText := punctuationAndEncodingReplacer.Replace(strings.ToLower(rawToken))
	return wordOrPunctuationPattern.FindAllString(lowercasedText, -1)
}

// RewriteTokenFileWithCleanedTokens reads a one-token-per-line file, normalizes
// it, and writes the result one token per line to the destination path.
func RewriteTokenFileWithCleanedTokens(sourcePath, destinationPath string) ([]string, error) {
	sourceFile, err := os.Open(sourcePath)
	if err != nil {
		return nil, err
	}

	var rawTokens []string
	scanner := bufio.NewScanner(sourceFile)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			rawTokens = append(rawTokens, line)
		}
	}
	err = scanner.Err()
	sourceFile.Close()
	if err != nil {
		return nil, err
	}

	var cleanedTokens []string
	for _, rawToken := range rawTokens {
		cleanedTokens = append(cleanedTokens, NormalizeAndSplitToken(rawToken)...)
	}

	destinationFile, err := os.Create(destinationPath)
	if err != nil {
		return nil, err
	}
	defer destinationFile.Close()

	writer := bufio.NewWriter(destinationFile)
	for _, token := range cleanedTokens {
		if _, err := fmt.Fprintln(writer, token); err != nil {
			return nil, err
		}
	}
	if err := writer.Flush(); err != nil {
		return nil, err
	}
	return cleanedTokens, nil
}
