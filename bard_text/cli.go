package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"time"
)

const (
	maxGeneratedWords = 40  // hard cap on how much the model will yap
	singleWordPartner = "." // paired with a lone word so there are always two
	cliTemperature    = 0.8 // safer than 1.0, so output loops slightly less
)

// RunCommandLineChat reads phrases from standard input and prints what the
// model says next. An empty line or Ctrl-D quits.
func RunCommandLineChat(model *LanguageModel, vocabulary *Vocabulary) {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Printf("type a phrase (max %d generated words). empty line or Ctrl-D quits.\n", maxGeneratedWords)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			fmt.Println()
			return
		}
		phrase := strings.TrimSpace(scanner.Text())
		if phrase == "" {
			return
		}
		respondToPhrase(model, vocabulary, phrase)
	}
}

// respondToPhrase takes the last two tokens of a phrase and generates from them.
func respondToPhrase(model *LanguageModel, vocabulary *Vocabulary, phrase string) {
	tokens := NormalizeAndSplitToken(phrase)
	if len(tokens) == 0 {
		fmt.Println("no words found in that phrase")
		return
	}

	// Only the last two tokens are used as context.
	if len(tokens) > 2 {
		tokens = tokens[len(tokens)-2:]
	}

	contextWordIDs := make([]int, 0, 2)
	for _, token := range tokens {
		wordID, exists := vocabulary.TokenToID(token)
		if !exists {
			fmt.Printf("%q is not in the vocabulary\n", token)
			return
		}
		contextWordIDs = append(contextWordIDs, wordID)
	}

	// A single word gets paired with "." so the model always has two.
	if len(contextWordIDs) == 1 {
		partnerID, exists := vocabulary.TokenToID(singleWordPartner)
		if !exists {
			fmt.Printf("cannot pair a single word: %q is not in the vocabulary\n", singleWordPartner)
			return
		}
		contextWordIDs = append(contextWordIDs, partnerID)
	}

	startWordIDs := [2]int{contextWordIDs[0], contextWordIDs[1]}
	randomSeed := time.Now().UnixNano()
	fmt.Println(GenerateTextFromStartWords(model, vocabulary, startWordIDs, maxGeneratedWords, cliTemperature, randomSeed))
}
