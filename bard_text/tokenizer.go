package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// Vocabulary maps tokens to IDs and IDs back to tokens.
type Vocabulary struct {
	tokenToID       map[string]int
	idToToken       []string
	totalTokenCount int
}

func NewVocabulary() *Vocabulary {
	return &Vocabulary{tokenToID: make(map[string]int)}
}

// AddTokenAndReturnID records one occurrence of a token and returns its ID.
func (vocabulary *Vocabulary) AddTokenAndReturnID(token string) int {
	vocabulary.totalTokenCount++
	// The token already exists.
	if existingID, exists := vocabulary.tokenToID[token]; exists {
		return existingID
	}
	newID := len(vocabulary.idToToken) // The next free ID.
	vocabulary.tokenToID[token] = newID
	vocabulary.idToToken = append(vocabulary.idToToken, token)
	return newID
}

// TokenToID returns the ID for a token, and whether it exists.
func (vocabulary *Vocabulary) TokenToID(token string) (int, bool) {
	id, exists := vocabulary.tokenToID[token]
	return id, exists
}

// IDToToken returns the token for an ID, and whether it exists.
func (vocabulary *Vocabulary) IDToToken(id int) (string, bool) {
	if id < 0 || id >= len(vocabulary.idToToken) {
		return "", false
	}
	return vocabulary.idToToken[id], true
}

// VocabularySize returns the number of distinct tokens.
func (vocabulary *Vocabulary) VocabularySize() int {
	return len(vocabulary.idToToken)
}

// TotalTokenCount returns the number of tokens, repeats included.
func (vocabulary *Vocabulary) TotalTokenCount() int {
	return vocabulary.totalTokenCount
}

// ReadTokenFileAsIDs reads one token per line and returns the ID of each token.
func ReadTokenFileAsIDs(filePath string, vocabulary *Vocabulary) ([]int, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var tokenIDs []int
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		tokenID, exists := vocabulary.TokenToID(line)
		if !exists {
			return nil, fmt.Errorf("token not in vocabulary: %q", line)
		}
		tokenIDs = append(tokenIDs, tokenID)
	}
	return tokenIDs, scanner.Err()
}

// LoadVocabularyFromFile reads one token per line and builds a vocabulary.
func LoadVocabularyFromFile(filePath string) (*Vocabulary, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	vocabulary := NewVocabulary()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		vocabulary.AddTokenAndReturnID(line)
	}
	return vocabulary, scanner.Err()
}
