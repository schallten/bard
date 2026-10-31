package main

import "fmt"

// TrainingExample is one training case: two context word IDs and the ID of the
// word that actually came next.
type TrainingExample struct {
	ContextWordIDs    [2]int // the two previous tokens
	CorrectNextWordID int    // the token that actually follows them
}

// BuildTrainingExamples slides a three-word window over the token IDs.
// Each window is one example: the first two IDs are context, the third is target.
func BuildTrainingExamples(tokenIDs []int) []TrainingExample {
	if len(tokenIDs) < 3 {
		return nil
	}
	trainingExamples := make([]TrainingExample, 0, len(tokenIDs)-2)
	for windowStartIndex := 0; windowStartIndex+2 < len(tokenIDs); windowStartIndex++ {
		example := TrainingExample{
			ContextWordIDs:    [2]int{tokenIDs[windowStartIndex], tokenIDs[windowStartIndex+1]},
			CorrectNextWordID: tokenIDs[windowStartIndex+2],
		}
		trainingExamples = append(trainingExamples, example)
	}
	return trainingExamples
}

// PrintTrainingExamplesAsWords prints examples so alignment can be checked by eye.
func PrintTrainingExamplesAsWords(vocabulary *Vocabulary, trainingExamples []TrainingExample, howManyToPrint int) {
	if howManyToPrint > len(trainingExamples) {
		howManyToPrint = len(trainingExamples)
	}
	for _, example := range trainingExamples[:howManyToPrint] {
		firstContextWord, _ := vocabulary.IDToToken(example.ContextWordIDs[0])
		secondContextWord, _ := vocabulary.IDToToken(example.ContextWordIDs[1])
		correctNextWord, _ := vocabulary.IDToToken(example.CorrectNextWordID)
		fmt.Printf("%q %q -> %q\n", firstContextWord, secondContextWord, correctNextWord)
	}
}
