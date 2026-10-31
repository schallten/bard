package main

import (
	"math/rand"
	"strings"
)

// SampleNextWordID draws one word ID from the model's predicted distribution.
// A temperature below 1 is safer, above 1 is wilder.
func (model *LanguageModel) SampleNextWordID(firstContextWordID, secondContextWordID int, temperature float32, randomSource *rand.Rand) int {
	logits := model.ComputeLogitsForEveryWord(firstContextWordID, secondContextWordID)
	if temperature != 1 {
		for candidateWordID := range logits {
			logits[candidateWordID] /= temperature
		}
	}
	probabilities := ConvertLogitsToProbabilities(logits)

	randomNumber := randomSource.Float32()
	var cumulativeProbability float32
	for candidateWordID, probability := range probabilities {
		cumulativeProbability += probability
		if randomNumber < cumulativeProbability {
			return candidateWordID
		}
	}
	return len(probabilities) - 1
}

// GenerateTextFromStartWords continues for a number of words from a two-word start.
func GenerateTextFromStartWords(model *LanguageModel, vocabulary *Vocabulary, startWordIDs [2]int, howManyWordsToGenerate int, temperature float32, randomSeed int64) string {
	randomSource := rand.New(rand.NewSource(randomSeed))
	generatedWordIDs := []int{startWordIDs[0], startWordIDs[1]}
	for wordIndex := 0; wordIndex < howManyWordsToGenerate; wordIndex++ {
		nextWordID := model.SampleNextWordID(
			generatedWordIDs[len(generatedWordIDs)-2],
			generatedWordIDs[len(generatedWordIDs)-1],
			temperature,
			randomSource)
		generatedWordIDs = append(generatedWordIDs, nextWordID)
	}

	var textBuilder strings.Builder
	for wordIndex, wordID := range generatedWordIDs {
		if wordIndex > 0 {
			textBuilder.WriteByte(' ')
		}
		word, _ := vocabulary.IDToToken(wordID)
		textBuilder.WriteString(word)
	}
	return textBuilder.String()
}
