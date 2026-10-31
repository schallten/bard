package main

import (
	"fmt"
	"log"
)

const (
	rawTokenFilePath     = "hamlet_tokens.txt"
	cleanedTokenFilePath = "hamlet_clean_tokens.txt"
)

func main() {
	// Lowercase and split punctuation off words, then load the result.
	if _, err := RewriteTokenFileWithCleanedTokens(rawTokenFilePath, cleanedTokenFilePath); err != nil {
		log.Fatal(err)
	}
	vocabulary, err := LoadVocabularyFromFile(cleanedTokenFilePath)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("total tokens: ", vocabulary.TotalTokenCount())
	fmt.Println("Vocabulary size:", vocabulary.VocabularySize())

	for wordID := 0; wordID < vocabulary.VocabularySize(); wordID++ {
		token, exists := vocabulary.IDToToken(wordID) // ID to token
		if !exists {
			log.Fatalf("decode failed for id %d", wordID)
		}
		roundTripID, exists := vocabulary.TokenToID(token) // token to ID
		if !exists || roundTripID != wordID {
			log.Fatalf("round trip failed for id %d (%q)", wordID, token)
		}
	}
	tokenIDs, err := ReadTokenFileAsIDs(cleanedTokenFilePath, vocabulary)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Encoded length:", len(tokenIDs))
	fmt.Println("First 10 IDs:", tokenIDs[:10])
	fmt.Println("ok")

	// Hold out the last 10% of the play for validation. The split is
	// contiguous, not random, so nearby sliding windows cannot leak
	// across the boundary.
	tokenSplitIndex := len(tokenIDs) * 9 / 10
	trainingExamples := BuildTrainingExamples(tokenIDs[:tokenSplitIndex])
	validationExamples := BuildTrainingExamples(tokenIDs[tokenSplitIndex:])
	fmt.Println("Train examples:", len(trainingExamples))
	fmt.Println("Val examples:  ", len(validationExamples))

	// Confirm every ID is in range, since the largest should be vocabulary size minus one.
	largestWordID := 0
	for _, wordID := range tokenIDs {
		if wordID > largestWordID {
			largestWordID = wordID
		}
	}
	fmt.Println("Max ID:", largestWordID)

	// Print the first few examples as words to check alignment.
	PrintTrainingExamplesAsWords(vocabulary, trainingExamples, 5)

	fmt.Println("-------------------------------")
	// Create a model with one weight per feature-candidate pair and a fixed seed.
	model := NewLanguageModel(vocabulary.VocabularySize(), 42)

	// Predict the word after the first training example's context.
	nextWordProbabilities := model.PredictNextWordProbabilities(
		trainingExamples[0].ContextWordIDs[0],
		trainingExamples[0].ContextWordIDs[1])

	// The probabilities should add up to about 1.
	var totalProbability float32
	for _, probability := range nextWordProbabilities {
		totalProbability += probability
	}
	fmt.Println("Sum of probabilities:", totalProbability) // Should print a value very close to 1.

	// Show how much probability the untrained model gives to the correct answer.
	fmt.Println("P(correct):", nextWordProbabilities[trainingExamples[0].CorrectNextWordID]) // Around 1/V, since weights are random.

	fmt.Println("-------------------------------")
	fmt.Printf("loss before training: train %.4f   val %.4f\n",
		model.AverageLossOverExamples(trainingExamples),
		model.AverageLossOverExamples(validationExamples))
	model.TrainModelForEpochs(trainingExamples, validationExamples, 10, 0.1)
	fmt.Printf("loss after training:  train %.4f   val %.4f\n",
		model.AverageLossOverExamples(trainingExamples),
		model.AverageLossOverExamples(validationExamples))

	CheckGradientAgainstNumericalEstimate(model, trainingExamples[0], 1e-2)

	fmt.Println("-------------------------------")
	RunCommandLineChat(model, vocabulary)
}
