package main

import (
	"math"
	"math/rand"
)

// LanguageModel holds every learnable number in the model.
type LanguageModel struct {
	VocabularySize int       // how many distinct words the model can predict
	Weights        []float32 // two blocks of rows; columns are candidate words
	Biases         []float32 // one baseline score per candidate word
}

// NewLanguageModel builds a model with small random weights and zero biases.
func NewLanguageModel(vocabularySize int, randomSeed int64) *LanguageModel {
	randomSource := rand.New(rand.NewSource(randomSeed))

	model := &LanguageModel{
		VocabularySize: vocabularySize,
		Weights:        make([]float32, 2*vocabularySize*vocabularySize),
		Biases:         make([]float32, vocabularySize),
	}

	for weightIndex := range model.Weights {
		model.Weights[weightIndex] = float32(randomSource.NormFloat64() * 0.01)
	}
	return model
}

// ComputeLogitsForEveryWord returns one raw score per candidate next word.
func (model *LanguageModel) ComputeLogitsForEveryWord(firstContextWordID, secondContextWordID int) []float32 {
	logits := make([]float32, model.VocabularySize)

	firstContextWordWeightRowStart := firstContextWordID * model.VocabularySize
	secondContextWordWeightRowStart := (model.VocabularySize + secondContextWordID) * model.VocabularySize

	for candidateWordID := 0; candidateWordID < model.VocabularySize; candidateWordID++ {
		logits[candidateWordID] = model.Biases[candidateWordID] +
			model.Weights[firstContextWordWeightRowStart+candidateWordID] +
			model.Weights[secondContextWordWeightRowStart+candidateWordID]
	}
	return logits
}

// ConvertLogitsToProbabilities turns raw scores into a distribution summing to 1.
func ConvertLogitsToProbabilities(logits []float32) []float32 {
	largestLogit := logits[0]
	for _, logit := range logits[1:] {
		if logit > largestLogit {
			largestLogit = logit
		}
	}

	probabilities := make([]float32, len(logits))
	var sumOfExponentials float32
	for candidateWordID, logit := range logits {
		exponential := float32(math.Exp(float64(logit - largestLogit)))
		probabilities[candidateWordID] = exponential
		sumOfExponentials += exponential
	}
	for candidateWordID := range probabilities {
		probabilities[candidateWordID] /= sumOfExponentials
	}
	return probabilities
}

// PredictNextWordProbabilities returns a probability for every candidate word.
func (model *LanguageModel) PredictNextWordProbabilities(firstContextWordID, secondContextWordID int) []float32 {
	return ConvertLogitsToProbabilities(model.ComputeLogitsForEveryWord(firstContextWordID, secondContextWordID))
}
