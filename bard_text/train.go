package main

import (
	"fmt"
	"math"
)

// LearnFromOneExample updates the model once using a single training example
// and returns the loss measured before the update.
func (model *LanguageModel) LearnFromOneExample(firstContextWordID, secondContextWordID, correctNextWordID int, learningRate float32) float32 {
	predictedProbabilities := model.PredictNextWordProbabilities(firstContextWordID, secondContextWordID)
	lossForThisExample := ComputeCrossEntropyLoss(predictedProbabilities, correctNextWordID)

	firstContextWordWeightRowStart := firstContextWordID * model.VocabularySize
	secondContextWordWeightRowStart := (model.VocabularySize + secondContextWordID) * model.VocabularySize

	for candidateWordID, predictedProbability := range predictedProbabilities {
		// The correction is guess minus truth, in one line.
		correction := predictedProbability
		if candidateWordID == correctNextWordID {
			correction -= 1
		}
		model.Biases[candidateWordID] -= learningRate * correction
		model.Weights[firstContextWordWeightRowStart+candidateWordID] -= learningRate * correction
		model.Weights[secondContextWordWeightRowStart+candidateWordID] -= learningRate * correction
	}
	return lossForThisExample
}

// TrainModelForEpochs sweeps the training examples and reports train and
// validation loss after every pass.
func (model *LanguageModel) TrainModelForEpochs(trainingExamples, validationExamples []TrainingExample, numberOfEpochs int, learningRate float32) {
	for epochNumber := 1; epochNumber <= numberOfEpochs; epochNumber++ {
		var sumOfLosses float64
		for _, example := range trainingExamples {
			sumOfLosses += float64(model.LearnFromOneExample(
				example.ContextWordIDs[0], example.ContextWordIDs[1], example.CorrectNextWordID, learningRate))
		}
		averageTrainingLoss := sumOfLosses / float64(len(trainingExamples))
		averageValidationLoss := model.AverageLossOverExamples(validationExamples)
		fmt.Printf("epoch %2d   train %.4f   val %.4f   val perplexity %.1f\n",
			epochNumber, averageTrainingLoss, averageValidationLoss, math.Exp(float64(averageValidationLoss)))
	}
}

// AverageLossOverExamples measures loss over all examples without changing the model.
func (model *LanguageModel) AverageLossOverExamples(examples []TrainingExample) float32 {
	var sumOfLosses float64
	for _, example := range examples {
		sumOfLosses += float64(ComputeCrossEntropyLoss(
			model.PredictNextWordProbabilities(example.ContextWordIDs[0], example.ContextWordIDs[1]),
			example.CorrectNextWordID))
	}
	return float32(sumOfLosses / float64(len(examples)))
}

// CheckGradientAgainstNumericalEstimate compares the analytic correction for two
// weights against a numerical one. Run it after some training, when logits are
// large enough to stand above float32 rounding noise.
func CheckGradientAgainstNumericalEstimate(model *LanguageModel, example TrainingExample, tinyWeightChange float32) {
	firstContextWordID := example.ContextWordIDs[0]
	secondContextWordID := example.ContextWordIDs[1]
	correctNextWordID := example.CorrectNextWordID
	predictedProbabilities := model.PredictNextWordProbabilities(firstContextWordID, secondContextWordID)
	firstContextWordWeightRowStart := firstContextWordID * model.VocabularySize

	measureLoss := func() float32 {
		return ComputeCrossEntropyLoss(
			model.PredictNextWordProbabilities(firstContextWordID, secondContextWordID),
			correctNextWordID)
	}

	for _, candidateWordID := range []int{correctNextWordID, (correctNextWordID + 1) % model.VocabularySize} {
		analyticCorrection := predictedProbabilities[candidateWordID]
		if candidateWordID == correctNextWordID {
			analyticCorrection -= 1
		}

		weightIndex := firstContextWordWeightRowStart + candidateWordID
		originalWeight := model.Weights[weightIndex]
		model.Weights[weightIndex] = originalWeight + tinyWeightChange
		lossAfterIncreasingWeight := measureLoss()
		model.Weights[weightIndex] = originalWeight - tinyWeightChange
		lossAfterDecreasingWeight := measureLoss()
		model.Weights[weightIndex] = originalWeight

		numericalCorrection := (lossAfterIncreasingWeight - lossAfterDecreasingWeight) / (2 * tinyWeightChange)
		fmt.Printf("Weights[%d][%d]  analytic %+.6f   numeric %+.6f\n",
			firstContextWordID, candidateWordID, analyticCorrection, numericalCorrection)
	}
}
